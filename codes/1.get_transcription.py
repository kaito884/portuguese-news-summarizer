import time
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# Caminho do arquivo CSV de entrada e saída
file_path = "../datas/1.youtube_data.csv"
output_file = "../datas/2.data_with_transcription.csv"

# Carregar os dados em um DataFrame
try:
    df = pd.read_csv(file_path)
    print(f"Arquivo '{file_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()

# Remover video_id repetido
print(f"Número original de entradas: {len(df)}")
df.drop_duplicates(subset=["video_id"], keep="first", inplace=True)
print(f"Número de entradas após remover video_ids duplicados: {len(df)}")


data_list = []# Lista para armazenar os vídeos com transcrição

def fetch_transcripts(records, category_id=25, search_limit=100, max_retries=3, retry_delay=5):
    """
    Filtra entradas, busca transcrições com mecanismo de retry.

    Args:
        records (list[dict]): lista de dicionários com os metadados dos vídeos.
        category_id (int): valor de categoryId a ser filtrado.
        search_limit (int): número máximo de entradas a processar.
        max_retries (int): número máximo de tentativas por vídeo.
        retry_delay (int): segundos de espera base entre tentativas.

    Returns:
        list[dict]: cada item contém os campos originais + 'transcription'.
    """
    transcripts = []
    processed_count = 0

    for i, entry in enumerate(records):
        if processed_count >= search_limit:
            print(f"\nLimite de busca ({search_limit}) atingido.")
            break

        video_id = entry.get("video_id")
        if not video_id or entry.get("categoryId") != category_id:
            continue

        processed_count += 1
        print(f"\nProcessando vídeo {processed_count}/{search_limit} (ID: {video_id})")

        # --- Início do Loop de Tentativas ---
        for attempt in range(max_retries):
            try:
                print(f"  -> Tentativa {attempt + 1}/{max_retries}...")
                
                # Obtém a transcrição
                lines = YouTubeTranscriptApi.get_transcript(video_id, languages=["pt-BR", "pt"])
                full_text = " ".join([t["text"] for t in lines])

                # Armazena transcrição e sai do loop de tentativas (sucesso!)
                item = entry.copy()
                item["transcription"] = full_text
                transcripts.append(item)
                print(f"    -> Transcrição encontrada para {video_id}.")
                break

            except (NoTranscriptFound, TranscriptsDisabled) as e:
                # Erros que nao devem ser tentados novamente
                print(f"    -> Transcrição não encontrada/desativada para {video_id}: {type(e).__name__}. (Não tentará novamente)")
                break

            except Exception as e:
                # Outros erros - tenta novamente
                print(f"    -> Erro na tentativa {attempt + 1}: {e}")
                
                # Se for a última tentativa, informa e desiste
                if attempt + 1 == max_retries:
                    print(f"    -> Limite de {max_retries} tentativas atingido. Desistindo de {video_id}.")
                else:
                    # espera um pouco e vai para proxima tentativa
                    wait_time = retry_delay * (attempt + 1)
                    print(f"    -> Aguardando {wait_time} segundos antes de tentar novamente...")
                    time.sleep(wait_time)
        # --- Fim do Loop de Tentativas ---

    print(f"\nTotal de transcrições obtidas: {len(transcripts)}")
    return transcripts

# Converte o DataFrame para uma lista de dicionários
records = df.to_dict(orient="records")

# 1) Buscar transcrições
data_list = fetch_transcripts(records, category_id=25, search_limit=4000, max_retries=3, retry_delay=0)

# 2) Salvar os dados no output_file
if data_list:
    df_with_transcripts = pd.DataFrame(data_list)
    try:
        df_with_transcripts.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nDados com transcrições salvos com sucesso em '{output_file}'.")
    except Exception as e:
        print(f"\nErro ao salvar o arquivo '{output_file}': {e}")
else:
    print("\nNenhuma transcrição foi obtida. Nenhum arquivo foi salvo.")