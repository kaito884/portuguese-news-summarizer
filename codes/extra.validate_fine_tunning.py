import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel
import math

#Compara o modelo sem fine-tunning e com fine-tunning

# --- 1. Configurações ---
VALIDATION_CSV_PATH = "../datas/4.validation_ideal_summary.csv"  # Seu arquivo CSV de validação
OUTPUT_COMPARISON_CSV_PATH = "../datas/6.evaluation_comparison_results.csv" # Arquivo de saída

TRANSCRIPTION_COLUMN = "transcription" # Nome da coluna com as transcrições
IDEAL_SUMMARY_COLUMN = "ideal_summary"   # Nome da coluna com os resumos ideais

BASE_MODEL_NAME = "recogna-nlp/ptt5-base-summ" # Modelo base usado para o fine-tuning
LORA_ADAPTER_DIR = "./ptt5_finetuned_lora_final" # Diretório onde o adaptador LoRA foi salvo

# Parâmetros de Geração (use os mesmos que você usaria no seu script generate_summary.py)
TASK_PREFIX = "resuma: " # Ou o prefixo que você usou no fine-tuning
GEN_MIN_LENGTH = 15
GEN_MAX_NEW_TOKENS = 150 # Use o mesmo max_new_tokens para ambos os modelos para uma comparação justa
GEN_NO_REPEAT_NGRAM_SIZE = 3
GEN_NUM_BEAMS = 4
GEN_EARLY_STOPPING = True
TOKEN_AUTH = None # Defina como True ou seu token string se o modelo base/LoRA exigir

# Limite de linhas para processar para teste rápido (None para todas as linhas do CSV de validação)
NUM_ROWS_TO_EVALUATE = None # Ex: 10 para testar com 10 linhas

MINI_BATCH_SIZE_EVAL = 8 # Tamanho do lote para geração (pode ser diferente do treino)
SAFE_TOKENIZER_INPUT_MAX_LENGTH = 1024 # Mesmo valor do script de treino/geração

# --- 2. Configuração do Dispositivo ---
DEVICE_IDX = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "cuda" if DEVICE_IDX == 0 else "cpu"
print(f"Usando dispositivo: {DEVICE_NAME.upper()}")

# --- 3. Funções Auxiliares ---
def load_model_and_tokenizer(model_name, adapter_path=None, token_auth=None):
    """Carrega o modelo (base ou com adaptador LoRA) e o tokenizador."""
    print(f"Carregando tokenizador de: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token_auth, use_fast=True)
    
    loaded_model_max_length = tokenizer.model_max_length
    actual_tokenizer_input_max_length = SAFE_TOKENIZER_INPUT_MAX_LENGTH
    if isinstance(loaded_model_max_length, int) and 0 < loaded_model_max_length <= (SAFE_TOKENIZER_INPUT_MAX_LENGTH * 4):
        actual_tokenizer_input_max_length = loaded_model_max_length
    tokenizer.model_max_length = actual_tokenizer_input_max_length
    print(f"  > tokenizer.model_max_length ajustado para: {tokenizer.model_max_length}")

    if hasattr(tokenizer, 'legacy'): # Para suprimir warning do T5Tokenizer
        tokenizer.legacy = False

    print(f"Carregando modelo base: {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token_auth)
    
    if adapter_path:
        print(f"Aplicando adaptador LoRA de: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Adaptador LoRA aplicado.")
    
    model = model.to(DEVICE_NAME)
    model.eval() # Colocar em modo de avaliação
    print("Modelo pronto.")
    return model, tokenizer

def generate_summaries_in_batches(texts, summarizer_pipeline_obj, batch_size, prefix, **gen_kwargs):
    """Gera resumos em mini-lotes para uma lista de textos."""
    all_summaries = []
    num_total_texts = len(texts)
    num_mini_batches = math.ceil(num_total_texts / batch_size)
    print(f"  Iniciando geração para {num_total_texts} textos em {num_mini_batches} mini-lotes de até {batch_size}...")

    for i in range(num_mini_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_total_texts)
        current_mini_batch_texts_raw = texts[start_index:end_index]
        current_mini_batch_with_prefix = [prefix + text for text in current_mini_batch_texts_raw]
        
        print(f"    Processando mini-lote {i+1}/{num_mini_batches} (textos {start_index+1}-{end_index})...")
        
        try:
            # A pipeline lida com a tokenização interna se passarmos strings
            pipeline_outputs = summarizer_pipeline_obj(
                current_mini_batch_with_prefix,
                **gen_kwargs
            )
            mini_batch_summaries = [item['summary_text'] for item in pipeline_outputs]
            all_summaries.extend(mini_batch_summaries)
        except Exception as e:
            print(f"    Erro durante a sumarização do mini-lote {i+1}: {e}")
            error_fill = ["Erro na geração do resumo" for _ in current_mini_batch_texts_raw]
            all_summaries.extend(error_fill)
            # import traceback # Para depuração mais profunda
            # traceback.print_exc()
    print("  Geração para este modelo concluída.")
    return all_summaries

# --- 4. Script Principal de Avaliação ---
if __name__ == "__main__":
    print(f"Carregando dados de validação de: {VALIDATION_CSV_PATH}...")
    try:
        df_val = pd.read_csv(VALIDATION_CSV_PATH)
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV de validação '{VALIDATION_CSV_PATH}' não encontrado.")
        exit()
    
    print(f"Dados de validação carregados. Total de linhas: {len(df_val)}")
    
    # Limitar número de linhas para avaliação, se definido
    if NUM_ROWS_TO_EVALUATE is not None and NUM_ROWS_TO_EVALUATE > 0 and NUM_ROWS_TO_EVALUATE < len(df_val):
        df_eval = df_val.head(NUM_ROWS_TO_EVALUATE).copy()
        print(f"Avaliando as primeiras {NUM_ROWS_TO_EVALUATE} linhas.")
    else:
        df_eval = df_val.copy()
        print(f"Avaliando todas as {len(df_eval)} linhas do arquivo de validação.")

    df_eval[TRANSCRIPTION_COLUMN] = df_eval[TRANSCRIPTION_COLUMN].fillna('')
    eval_transcriptions = df_eval[TRANSCRIPTION_COLUMN].tolist()

    # Parâmetros de geração comuns para ambos os modelos
    generation_params = {
        "min_length": GEN_MIN_LENGTH,
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "no_repeat_ngram_size": GEN_NO_REPEAT_NGRAM_SIZE,
        "num_beams": GEN_NUM_BEAMS,
        "early_stopping": GEN_EARLY_STOPPING,
        "do_sample": False,
        "truncation": True # A pipeline cuida da truncação da entrada
    }

    # --- 4.1 Gerar Resumos com o Modelo Base Original ---
    print("\n--- Gerando resumos com o MODELO BASE ORIGINAL ---")
    base_model_obj, base_tokenizer_obj = load_model_and_tokenizer(BASE_MODEL_NAME, token_auth=TOKEN_AUTH)
    # Criar uma pipeline para o modelo base para usar a mesma interface de batching
    base_summarizer_pipeline = pipeline(
        "summarization", 
        model=base_model_obj, 
        tokenizer=base_tokenizer_obj, 
        device=DEVICE_IDX
    )
    df_eval["summary_base_model"] = generate_summaries_in_batches(
        eval_transcriptions, base_summarizer_pipeline, MINI_BATCH_SIZE_EVAL, TASK_PREFIX, **generation_params
    )

    # --- 4.2 Gerar Resumos com o Modelo Fine-Tuned (LoRA) ---
    print("\n--- Gerando resumos com o MODELO FINE-TUNED (LoRA) ---")
    # O tokenizer é o mesmo do modelo base
    finetuned_model_obj, _ = load_model_and_tokenizer(BASE_MODEL_NAME, adapter_path=LORA_ADAPTER_DIR, token_auth=TOKEN_AUTH)
    # Criar uma pipeline para o modelo fine-tuned
    finetuned_summarizer_pipeline = pipeline(
        "summarization", 
        model=finetuned_model_obj, 
        tokenizer=base_tokenizer_obj, # Reutilizar o tokenizer do modelo base
        device=DEVICE_IDX
    )
    df_eval["summary_finetuned_lora"] = generate_summaries_in_batches(
        eval_transcriptions, finetuned_summarizer_pipeline, MINI_BATCH_SIZE_EVAL, TASK_PREFIX, **generation_params
    )
    
    # --- 5. Organizar e Salvar Resultados da Comparação ---
    colunas_para_output = [
        TRANSCRIPTION_COLUMN, 
        IDEAL_SUMMARY_COLUMN, 
        "summary_base_model", 
        "summary_finetuned_lora"
    ]
    # Adicionar video_id e title se existirem no df_eval para melhor identificação
    if 'video_id' in df_eval.columns:
        colunas_para_output.insert(0, 'video_id')
    if 'title' in df_eval.columns:
        colunas_para_output.insert(1, 'title')
    
    # Garantir que todas as colunas selecionadas existem
    colunas_existentes_para_output = [col for col in colunas_para_output if col in df_eval.columns]

    df_comparison = df_eval[colunas_existentes_para_output]

    print(f"\nSalvando resultados da comparação em: {OUTPUT_COMPARISON_CSV_PATH}...")
    try:
        df_comparison.to_csv(OUTPUT_COMPARISON_CSV_PATH, index=False, encoding='utf-8')
        print("Resultados da comparação salvos com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar o arquivo CSV de comparação: {e}")

    print("\n--- Amostra dos Resultados da Comparação (primeiras 5 linhas) ---")
    if not df_comparison.empty:
        pd.set_option('display.max_colwidth', 100) # Ajustar largura para visualização
        print(df_comparison.head())
    else:
        print("Nenhum dado na comparação para mostrar.")
    print("-------------------------------------------------------------------\n")
    
    print("Script de avaliação finalizado.")