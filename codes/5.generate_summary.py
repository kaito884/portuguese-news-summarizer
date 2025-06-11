import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import math

# --- 1. Configurações e Parâmetros Essenciais ---
# Arquivo principal com transcrições, video_id, title
MAIN_DATA_CSV_PATH = '../datas/4.data_preprocessed.csv'
# Arquivo com as transcrições e os resumos ideais
IDEAL_SUMMARIES_CSV_PATH = '../datas/5.ideal_summary_all.csv'
# Coluna de resumo ideal no arquivo IDEAL_SUMMARIES_CSV_PATH
IDEAL_SUMMARY_COLUMN_NAME_IN_FILE = "ideal_summary"

OUTPUT_CSV_FILE_PATH = '../datas/6.data_with_summaries.csv' # Nome do arquivo de saída final
TRANSCRIPTION_COLUMN = 'transcription' # Nome da coluna de transcrição (comum a ambos os arquivos)
# Nome da coluna para os resumos gerados pelo modelo neste script
GENERATED_SUMMARY_COLUMN_NAME = 'summary_model_finetuned'

# --- Configurações do Modelo Fine-Tuned ---
BASE_MODEL_NAME = "recogna-nlp/ptt5-base-summ" 
LORA_ADAPTER_PATH = "../ptt5_finetuned_lora_final"

# Parâmetros de geração para o lote
TASK_PREFIX = "resuma: " 
BATCH_MIN_LENGTH = 15
BATCH_MAX_NEW_TOKENS = 150 
NO_REPEAT_NGRAM_SIZE = 3
NUM_BEAMS = 4
EARLY_STOPPING = True
TOKEN_AUTH = None 

NUM_ROWS_TO_PROCESS = None
SAFE_TOKENIZER_INPUT_MAX_LENGTH = 1024
MINI_BATCH_SIZE = 64

# --- 2. Configuração do Dispositivo ---
device_idx = 0 if torch.cuda.is_available() else -1
device_name = "cuda" if device_idx == 0 else "cpu"
print(f"Usando dispositivo: {device_name.upper()}")

# --- 3. Carregar Modelo Fine-Tuned (Base + LoRA) e Tokenizador ---
print(f"Carregando modelo base '{BASE_MODEL_NAME}' e aplicando adaptador LoRA de '{LORA_ADAPTER_PATH}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=TOKEN_AUTH, use_fast=True)
    
    loaded_model_max_length = tokenizer.model_max_length
    actual_tokenizer_input_max_length = SAFE_TOKENIZER_INPUT_MAX_LENGTH
    if isinstance(loaded_model_max_length, int) and 0 < loaded_model_max_length <= (SAFE_TOKENIZER_INPUT_MAX_LENGTH * 4):
        actual_tokenizer_input_max_length = loaded_model_max_length
    tokenizer.model_max_length = actual_tokenizer_input_max_length
    print(f"  > tokenizer.model_max_length ajustado para: {tokenizer.model_max_length}")

    if hasattr(tokenizer, 'legacy'):
        tokenizer.legacy = False

    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME, token=TOKEN_AUTH)
    model_to_use = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model_to_use = model_to_use.to(device_name)
    model_to_use.eval() 
    print("Modelo fine-tuned com LoRA carregado com sucesso!")

    summarizer_pipeline = pipeline(
        "summarization",
        model=model_to_use, 
        tokenizer=tokenizer, 
        device=device_idx 
    )
    print("Pipeline de sumarização com modelo fine-tuned pronta.")

except FileNotFoundError as fnf_error:
    print(f"Erro ao carregar modelo ou adaptador: Arquivo não encontrado.")
    print(f"BASE_MODEL_NAME: {BASE_MODEL_NAME}")
    print(f"LORA_ADAPTER_PATH: {LORA_ADAPTER_PATH}")
    print(f"Detalhes: {fnf_error}")
    exit()
except Exception as e:
    print(f"Erro ao carregar o modelo/tokenizador ou aplicar LoRA: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 4. Carregar e Preparar Dados ---
print(f"Carregando dados principais de: {MAIN_DATA_CSV_PATH}...")
try:
    df_main = pd.read_csv(MAIN_DATA_CSV_PATH)
    if TRANSCRIPTION_COLUMN not in df_main.columns:
        raise ValueError(f"Coluna '{TRANSCRIPTION_COLUMN}' não encontrada em '{MAIN_DATA_CSV_PATH}'")
except FileNotFoundError:
    print(f"Erro: Arquivo CSV principal '{MAIN_DATA_CSV_PATH}' não encontrado.")
    exit()
except ValueError as ve:
    print(f"Erro no arquivo principal: {ve}")
    exit()

print(f"Carregando resumos ideais de: {IDEAL_SUMMARIES_CSV_PATH}...")
try:
    df_ideal = pd.read_csv(IDEAL_SUMMARIES_CSV_PATH)
    if TRANSCRIPTION_COLUMN not in df_ideal.columns or IDEAL_SUMMARY_COLUMN_NAME_IN_FILE not in df_ideal.columns:
        raise ValueError(f"Colunas '{TRANSCRIPTION_COLUMN}' ou '{IDEAL_SUMMARY_COLUMN_NAME_IN_FILE}' não encontradas em '{IDEAL_SUMMARIES_CSV_PATH}'")
except FileNotFoundError:
    print(f"Erro: Arquivo CSV de resumos ideais '{IDEAL_SUMMARIES_CSV_PATH}' não encontrado.")
    exit()
except ValueError as ve:
    print(f"Erro no arquivo de resumos ideais: {ve}")
    exit()

# Juntar os DataFrames pela coluna de transcrição
print("Juntando dados principais com resumos ideais...")
df_merged = pd.merge(df_main, df_ideal[[TRANSCRIPTION_COLUMN, IDEAL_SUMMARY_COLUMN_NAME_IN_FILE]], on=TRANSCRIPTION_COLUMN, how="left")

# Verificar se houve correspondências
if df_merged[IDEAL_SUMMARY_COLUMN_NAME_IN_FILE].isna().any():
    print(f"AVISO: Algumas transcrições em '{MAIN_DATA_CSV_PATH}' não encontraram correspondência em '{IDEAL_SUMMARIES_CSV_PATH}'.")
    print(f"Número de NaNs na coluna '{IDEAL_SUMMARY_COLUMN_NAME_IN_FILE}': {df_merged[IDEAL_SUMMARY_COLUMN_NAME_IN_FILE].isna().sum()}")

print(f"Dados carregados e juntados. Total de linhas a serem consideradas: {len(df_merged)}")

df_to_process = df_merged.copy()
if NUM_ROWS_TO_PROCESS is not None and NUM_ROWS_TO_PROCESS > 0 and NUM_ROWS_TO_PROCESS < len(df_merged):
    df_to_process = df_merged.head(NUM_ROWS_TO_PROCESS).copy()
    print(f"Processando as primeiras {NUM_ROWS_TO_PROCESS} linhas para teste.")
else:
    if NUM_ROWS_TO_PROCESS is None or NUM_ROWS_TO_PROCESS >= len(df_merged):
         print(f"Processando todas as {len(df_to_process)} linhas.")
    else: 
        print("NUM_ROWS_TO_PROCESS configurado para 0 ou valor inválido. Nenhum resumo será gerado.")
        df_to_process = pd.DataFrame(columns=df_merged.columns) 

df_to_process.loc[:, TRANSCRIPTION_COLUMN] = df_to_process[TRANSCRIPTION_COLUMN].fillna('') # Garante que é string
transcription_texts_raw = df_to_process[TRANSCRIPTION_COLUMN].tolist()
transcription_texts_with_prefix = [TASK_PREFIX + text for text in transcription_texts_raw]

# --- 5. Geração de Resumos em Mini-Lotes com Modelo Fine-Tuned ---
all_generated_summaries_text = [] 

if transcription_texts_with_prefix:
    num_total_texts = len(transcription_texts_with_prefix)
    num_mini_batches = math.ceil(num_total_texts / MINI_BATCH_SIZE)
    print(f"\nIniciando geração de resumos para {num_total_texts} transcrições com o modelo fine-tuned, em {num_mini_batches} mini-lotes de até {MINI_BATCH_SIZE} cada...")

    for i in range(num_mini_batches):
        start_index = i * MINI_BATCH_SIZE
        end_index = min((i + 1) * MINI_BATCH_SIZE, num_total_texts)
        current_mini_batch_texts = transcription_texts_with_prefix[start_index:end_index]
        
        print(f"  Processando mini-lote {i+1}/{num_mini_batches} (textos {start_index+1} a {end_index} de {num_total_texts})...")
        
        if not current_mini_batch_texts:
            continue

        try:
            pipeline_outputs = summarizer_pipeline(
                current_mini_batch_texts,
                min_length=BATCH_MIN_LENGTH,
                max_new_tokens=BATCH_MAX_NEW_TOKENS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                num_beams=NUM_BEAMS,
                early_stopping=EARLY_STOPPING,
                do_sample=False,
                truncation=True 
            )
            mini_batch_summaries = [item['summary_text'] for item in pipeline_outputs]
            all_generated_summaries_text.extend(mini_batch_summaries)
        except Exception as e:
            print(f"  Erro durante a sumarização do mini-lote {i+1}: {e}")
            error_summaries = ["Erro durante a sumarização." for _ in current_mini_batch_texts]
            all_generated_summaries_text.extend(error_summaries)
    print("Geração de resumos concluída.")
else:
    print("\nNenhuma transcrição para processar.")

# Adicionar coluna de resumo gerado
if len(all_generated_summaries_text) == len(df_to_process):
    df_to_process[GENERATED_SUMMARY_COLUMN_NAME] = all_generated_summaries_text
elif not transcription_texts_with_prefix and len(df_to_process) > 0:
    df_to_process[GENERATED_SUMMARY_COLUMN_NAME] = ["Nenhuma transcrição fornecida" for _ in range(len(df_to_process))]
elif len(df_to_process) > 0 : 
    print(f"Alerta: Número de resumos gerados ({len(all_generated_summaries_text)}) não corresponde ao número de linhas ({len(df_to_process)}). Preenchendo com erro.")
    df_to_process[GENERATED_SUMMARY_COLUMN_NAME] = ["Erro ou desalinhamento no processamento" for _ in range(len(df_to_process))]
else: # df_to_process é vazio
    if GENERATED_SUMMARY_COLUMN_NAME not in df_to_process.columns:
         df_to_process[GENERATED_SUMMARY_COLUMN_NAME] = pd.Series(dtype='object')


# --- 6. Selecionar Colunas Finais e Salvar Resultados ---
FINAL_COLUMNS_ORDER = [
    'video_id', 
    'title', 
    TRANSCRIPTION_COLUMN, 
    IDEAL_SUMMARY_COLUMN_NAME_IN_FILE, # Esta é a coluna que veio do merge
    GENERATED_SUMMARY_COLUMN_NAME      # Esta é a coluna com os resumos que acabamos de gerar
]

# Filtrar para manter apenas as colunas desejadas que existem no DataFrame
df_output = df_to_process[[col for col in FINAL_COLUMNS_ORDER if col in df_to_process.columns]].copy()

# Renomear a coluna do resumo gerado para 'summary' se desejar um nome final específico
# Isso é opcional. Se você quiser que a coluna de resumo gerado se chame 'summary':
df_output.rename(columns={GENERATED_SUMMARY_COLUMN_NAME: 'summary'}, inplace=True)
# Se você renomeou, ajuste FINAL_COLUMNS_ORDER para a impressão de amostra, se necessário, ou a lista de impressão.

print(f"\nSalvando resultados em: {OUTPUT_CSV_FILE_PATH}...")
try:
    df_output.to_csv(OUTPUT_CSV_FILE_PATH, index=False, encoding='utf-8')
    print("Resultados salvos com sucesso!")
except Exception as e:
    print(f"Erro ao salvar o arquivo CSV: {e}")

# --- 7. Mostrar Amostra dos Resultados ---
print("\n--- Amostra dos Resultados (Colunas Finais) ---")
if not df_output.empty:
    # Mostrar as colunas na ordem definida, mas após o rename se ele ocorreu
    cols_to_show_final = ['video_id', 'title', TRANSCRIPTION_COLUMN, IDEAL_SUMMARY_COLUMN_NAME_IN_FILE, 'summary']
    cols_to_show_final_existing = [col for col in cols_to_show_final if col in df_output.columns]
    print(df_output[cols_to_show_final_existing].head())
else:
    print("Nenhum dado processado ou DataFrame vazio para mostrar.")
print("-----------------------------------------------\n")
print("Processo finalizado.")