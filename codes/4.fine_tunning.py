import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import evaluate
import nltk
import numpy as np

# --- 1. Configurações Essenciais ---
MODEL_NAME = "recogna-nlp/ptt5-base-summ"
TRAIN_CSV_FILE_PATH = "../datas/5.ideal_summary_train.csv"
VALIDATION_CSV_FILE_PATH = "../datas/5.ideal_summary_validation.csv"
TRANSCRIPTION_COLUMN = "transcription"
SUMMARY_COLUMN = "ideal_summary"
OUTPUT_DIR = "../ptt5_finetuned_lora_final" 
LOGGING_DIR = "../logs_finetuning_final"
TASK_PREFIX = "resuma: "
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150

# Parâmetros de Geração para model.generation_config (usados em trainer.evaluate)
EVAL_GENERATION_MIN_LENGTH = 15
EVAL_GENERATION_MAX_NEW_TOKENS = 150
EVAL_GENERATION_NUM_BEAMS = 4
EVAL_NO_REPEAT_NGRAM_SIZE = 3

# Parâmetros LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q", "v"]

# Hiperparâmetros de Treinamento
NUM_TRAIN_EPOCHS = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2
LOGGING_STEPS = 10 # Ajuste conforme o número de passos por época
SAVE_AND_EVAL_STEPS = 50 
LOAD_IN_8BIT = False

# --- Download NLTK Punkt --- (nao consegui fazer o download, mas deixo aqui a tentativa)
NLTK_PUNKT_AVAILABLE = False
try:
    print("Verificando/baixando recurso 'punkt' do NLTK (necessário para ROUGE-L)...")
    nltk.download("punkt", quiet=True)
    nltk.sent_tokenize("Teste. Teste.", language='portuguese')
    print("Recurso 'punkt' do NLTK para português parece estar funcional.")
    NLTK_PUNKT_AVAILABLE = True
except LookupError:
    print("AVISO: Recurso NLTK 'punkt' para português (ou seus componentes) não encontrado.")
    print("O cálculo de ROUGE-L pode ser afetado. ROUGE-N ainda deve funcionar.")
except Exception as e_nltk:
    print(f"AVISO: Erro ao tentar inicializar NLTK punkt: {e_nltk}")

# --- 2. Carregar e Preparar o Dataset ---
print("Carregando datasets de treino e validação...")
validation_dataset_available = False 
try:
    df_train = pd.read_csv(TRAIN_CSV_FILE_PATH)
    if df_train.empty or not (TRANSCRIPTION_COLUMN in df_train.columns and SUMMARY_COLUMN in df_train.columns):
        raise ValueError(f"Dataset de Treino '{TRAIN_CSV_FILE_PATH}' está vazio ou não contém as colunas necessárias.")
    df_train.dropna(subset=[TRANSCRIPTION_COLUMN, SUMMARY_COLUMN], inplace=True)
    df_train = df_train[df_train[TRANSCRIPTION_COLUMN].str.strip() != '']
    df_train = df_train[df_train[SUMMARY_COLUMN].str.strip() != '']
    print(f"Número de amostras válidas no dataset de Treino: {len(df_train)}")
    if len(df_train) < 1:
        raise ValueError("Dataset de Treino ficou vazio após limpeza.")

    try:
        df_validation = pd.read_csv(VALIDATION_CSV_FILE_PATH)
        if df_validation.empty or not (TRANSCRIPTION_COLUMN in df_validation.columns and SUMMARY_COLUMN in df_validation.columns):
            validation_dataset_available = False
        else:
            df_validation.dropna(subset=[TRANSCRIPTION_COLUMN, SUMMARY_COLUMN], inplace=True)
            df_validation = df_validation[df_validation[TRANSCRIPTION_COLUMN].str.strip() != '']
            df_validation = df_validation[df_validation[SUMMARY_COLUMN].str.strip() != '']
            if len(df_validation) < 1:
                validation_dataset_available = False
            else:
                print(f"Número de amostras válidas no dataset de Validação: {len(df_validation)}")
                validation_dataset_available = True
    except FileNotFoundError:
        validation_dataset_available = False
        df_validation = pd.DataFrame() 

    dataset_dict_constructor = {'train': Dataset.from_pandas(df_train)}
    if validation_dataset_available and not df_validation.empty:
        dataset_dict_constructor['validation'] = Dataset.from_pandas(df_validation)
        print(f"Datasets carregados: {len(dataset_dict_constructor['train'])} treino, {len(dataset_dict_constructor['validation'])} validação")
    else:
        print(f"Datasets carregados: {len(dataset_dict_constructor['train'])} treino. VALIDAÇÃO NÃO SERÁ USADA ou dataset de validação está vazio/inválido.")
        validation_dataset_available = False 
    
    dataset_dict = DatasetDict(dataset_dict_constructor)

except FileNotFoundError as fnf_error:
    print(f"Erro Crítico: Arquivo CSV não encontrado: {fnf_error}")
    exit()
except ValueError as ve:
    print(f"Erro Crítico nos dados: {ve}")
    exit()
except Exception as e:
    print(f"Erro crítico ao carregar os datasets: {e}")
    exit()

# --- 3. Tokenização ---
print("Carregando tokenizador e tokenizando os datasets...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if hasattr(tokenizer, 'legacy'): 
    tokenizer.legacy = False

def preprocess_function(examples):
    inputs = [TASK_PREFIX + doc for doc in examples[TRANSCRIPTION_COLUMN]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[SUMMARY_COLUMN], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

try:
    columns_to_remove = list(dataset_dict["train"].column_names)
    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    print("Tokenização concluída.")
except Exception as e:
    print(f"Erro durante a tokenização: {e}")
    exit()

# --- 4. Carregar Modelo Base e Aplicar LoRA ---
print("Carregando modelo base e aplicando configuração LoRA...")
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, load_in_8bit=LOAD_IN_8BIT, device_map="auto" if LOAD_IN_8BIT else None)
    if LOAD_IN_8BIT:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Ajustar o generation_config do modelo para influenciar trainer.evaluate()
    if model.generation_config is None:
        model.generation_config = GenerationConfig() 
    model.generation_config.min_length = EVAL_GENERATION_MIN_LENGTH
    model.generation_config.max_new_tokens = EVAL_GENERATION_MAX_NEW_TOKENS
    model.generation_config.num_beams = EVAL_GENERATION_NUM_BEAMS
    model.generation_config.early_stopping = True 
    model.generation_config.no_repeat_ngram_size = EVAL_NO_REPEAT_NGRAM_SIZE
    model.generation_config.do_sample = False
    print("Modelo com LoRA e generation_config para avaliação ajustados.")

except Exception as e:
    print(f"Erro ao carregar o modelo, aplicar LoRA ou ajustar generation_config: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- Função para Calcular Métricas ROUGE ---
metric = evaluate.load("rouge")
def compute_metrics_rouge(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds_cleaned = [pred.strip() for pred in decoded_preds if pred.strip()]
    decoded_labels_cleaned = [label.strip() for label in decoded_labels if label.strip()]

    if not decoded_preds_cleaned or not decoded_labels_cleaned or len(decoded_preds_cleaned) != len(decoded_labels_cleaned):
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0, "gen_len": 0.0}

    if NLTK_PUNKT_AVAILABLE:
        try:
            decoded_preds_for_rouge = ["\n".join(nltk.sent_tokenize(pred, language='portuguese')) for pred in decoded_preds_cleaned]
            decoded_labels_for_rouge = ["\n".join(nltk.sent_tokenize(label, language='portuguese')) for label in decoded_labels_cleaned]
        except LookupError:
            # Aviso já foi dado no início se NLTK_PUNKT_AVAILABLE é True mas falha aqui
            decoded_preds_for_rouge = decoded_preds_cleaned
            decoded_labels_for_rouge = decoded_labels_cleaned
    else:
        decoded_preds_for_rouge = decoded_preds_cleaned
        decoded_labels_for_rouge = decoded_labels_cleaned

    result = metric.compute(predictions=decoded_preds_for_rouge, references=decoded_labels_for_rouge, use_stemmer=True, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"])
    
    result_final = {}
    for key, value in result.items():
        result_final[key] = round(value * 100, 4)

    prediction_lens = [np.count_nonzero(pred_ids != tokenizer.pad_token_id) for pred_ids in predictions]
    result_final["gen_len"] = round(np.mean(prediction_lens), 4) if prediction_lens else 0.0
    
    return result_final

# --- 5. Configurar Argumentos de Treinamento e Trainer ---
print("Configurando argumentos de treinamento...")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else None)

use_bf16 = False
use_fp16 = False
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        use_bf16 = True
        print("  > BF16 é suportado e será usado.")
    else:
        use_fp16 = True
        print("  > BF16 não é suportado. Usando FP16.")

current_do_eval = validation_dataset_available
current_eval_steps = SAVE_AND_EVAL_STEPS if validation_dataset_available else None
current_save_steps = SAVE_AND_EVAL_STEPS
current_load_best_model_at_end = False 
current_metric_for_best_model = "loss" if validation_dataset_available else None
current_greater_is_better = False if validation_dataset_available else None

if validation_dataset_available:
    print(f"  > Avaliação habilitada: do_eval={current_do_eval}, eval_steps={current_eval_steps}")
    print(f"  > load_best_model_at_end está definido como: {current_load_best_model_at_end}")
else:
    print("  > Avaliação DESABILITADA.")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE if current_do_eval else PER_DEVICE_TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    logging_dir=LOGGING_DIR,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=current_save_steps,
    save_total_limit=3,
    do_eval=current_do_eval,
    eval_steps=current_eval_steps,
    bf16=use_bf16,
    fp16=use_fp16 and not use_bf16,
    load_best_model_at_end=current_load_best_model_at_end,
    metric_for_best_model=current_metric_for_best_model,
    greater_is_better=current_greater_is_better,
    remove_unused_columns=False,
    predict_with_generate=True if current_do_eval else False,
    # Os parâmetros de geração para `evaluate` serão pegos de `model.generation_config`
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_rouge if current_do_eval else None,
)

# --- 6. Treinar o Modelo ---
print("Iniciando o treinamento LoRA...")
try:
    train_result = trainer.train()
    print("Treinamento concluído.")

    print(f"Salvando o adaptador LoRA treinado em {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    if tokenizer is not None:
        tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adaptador LoRA e tokenizador salvos em {OUTPUT_DIR}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if validation_dataset_available and current_do_eval:
        print("\nExecutando avaliação final no conjunto de validação...")
        eval_metrics = trainer.evaluate()
        print("\nMétricas de Avaliação Final:")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

except Exception as e:
    print(f"Erro durante o treinamento: {e}")
    import traceback
    traceback.print_exc()

print("Script de fine-tuning finalizado.")