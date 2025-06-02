import pandas as pd
import re

# --- Configurações ---
# Arquivo de entrada
INPUT_CSV_FILE_PATH = '../datas/3.filtered_data.csv'
# Arquivo de saída com as transcrições pré-processadas
OUTPUT_CSV_FILE_PATH = '../datas/4.data_preprocessed.csv'
TRANSCRIPTION_COLUMN = 'transcription'

# --- Lista de Palavras/Frases de Preenchimento Comuns (ajuste conforme necessário) ---
# \b denota fronteira de palavra para pegar a palavra exata.
FILLER_PATTERNS_TO_REMOVE = [
    r'\b(e aí)\b',
    r'\b(né)\b',
    r'\b(tipo assim)\b',
    r'\b(então)\b', # "então" pode ser importante, use com cautela ou remova da lista
    r'\b(assim)\b',  # "assim" também, use com cautela
    r'\b(tipo)\b',   # Cuidado, "tipo de" pode ser importante
    r'\b(ahn?)\b', r'\b(ah?)\b', r'\b(eh?)\b',
    r'\b(hmm)\b', r'\b(hum)\b',
    r'\[música\]',
    r'\[aplausos\]',
    r'\[risadas\]',
    # Adicione outros padrões que você observa com frequência e que são ruído
]

def preprocess_text(text):
    """
    Aplica várias etapas de pré-processamento a um texto de transcrição.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Converter para minúsculas
    processed_text = text.lower() 

    # 2. Remover padrões de preenchimento e marcadores
    for pattern in FILLER_PATTERNS_TO_REMOVE:
        processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)

    # 3. Remover pontuação excessiva ou específica (opcional, use com cuidado)
    # Exemplo: remover múltiplos pontos de exclamação/interrogação
    # processed_text = re.sub(r'!{2,}', '!', processed_text)
    # processed_text = re.sub(r'\?{2,}', '?', processed_text)

    # 4. Normalizar espaços em branco
    # Remover espaços no início/fim e substituir múltiplos espaços por um único
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    # 5. Remover "e aí" especificamente no final da string, se ainda existir
    if processed_text.endswith(" e aí"):
      processed_text = processed_text[:-4].strip()
    if processed_text.endswith(" aí"): # Caso tenha sobrado só " aí"
      processed_text = processed_text[:-3].strip()


    # 6. (Opcional) Remover frases muito curtas ou apenas com restos de fillers
    # Isso é mais complexo e pode exigir análise de palavras.
    # Por exemplo, se após a limpeza a frase ficou com menos de 3 palavras, pode ser descartada
    # ou tratada de forma especial.

    return processed_text

# --- Script Principal ---
if __name__ == "__main__":
    print(f"Iniciando pré-processamento do arquivo: {INPUT_CSV_FILE_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_FILE_PATH)
        print(f"  Dados carregados. Total de linhas: {len(df)}")

        if TRANSCRIPTION_COLUMN not in df.columns:
            print(f"  Erro: A coluna '{TRANSCRIPTION_COLUMN}' não foi encontrada no CSV.")
            exit()

        # Aplicar a função de pré-processamento na coluna de transcrição
        df[TRANSCRIPTION_COLUMN] = df[TRANSCRIPTION_COLUMN].apply(preprocess_text)
        print(f"  Pré-processamento da coluna '{TRANSCRIPTION_COLUMN}' concluído.")

        # Salvar o DataFrame com as transcrições processadas
        df.to_csv(OUTPUT_CSV_FILE_PATH, index=False, encoding='utf-8')
        print(f"  Arquivo pré-processado salvo em: {OUTPUT_CSV_FILE_PATH}")

        # Mostrar algumas amostras
        print("\n--- Amostra das Transcrições Pré-processadas (primeiras 5 linhas) ---")
        if not df.empty:
            pd.set_option('display.max_colwidth', 200)
            print(df[[TRANSCRIPTION_COLUMN]].head())
        else:
            print("  DataFrame vazio após processamento.")
        print("------------------------------------------------------------------\n")

    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo de entrada '{INPUT_CSV_FILE_PATH}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o pré-processamento: {e}")

    print("Pré-processamento finalizado.")