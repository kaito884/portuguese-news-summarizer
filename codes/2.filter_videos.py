import pandas as pd
import re

input_filename = '../datas/2.data_with_transcription.csv' # arquivo de entrada antes da filtragem
output_filename = '../datas/3.filtered_data.csv' # arquivo de saída final

# 1. Definir os atributos a serem mantidos
colunas_desejadas = [
    'video_id', 'title', 'channelId', 'channelTitle', 'transcription', 'description'
]

# Limite de palavras para a transcrição
limite_palavras_transcricao = 1024

# Limite para a frequência da palavra "eu" (ex: 0.05 = 5%)
# Se mais de 5% das palavras forem "eu", o vídeo é filtrado. Ajuste conforme necessário.
# Colocado, pois geralmente os videos com muito `eu` sao mais dificeis de transcrever
limite_frequencia_eu = 0.01 

try:
    # Carregar os dados do CSV.
    df = pd.read_csv(input_filename)
    print(f"Linhas carregadas do arquivo '{input_filename}': {len(df)}")

    # --- Tarefa 1: Deixar somente os atributos especificados ---
    colunas_faltantes = [col for col in colunas_desejadas if col not in df.columns]
    if colunas_faltantes:
        raise KeyError(f"As seguintes colunas não foram encontradas no CSV: {', '.join(colunas_faltantes)}")

    df_selecionado = df[colunas_desejadas].copy()

    # Lidar com valores ausentes (NaN) na coluna 'transcription'
    df_selecionado.loc[:, 'transcription'] = df_selecionado['transcription'].fillna('')

    # --- Tarefa 2: Filtrar vídeos com número de palavras da transcrição > limite_palavras_transcricao ---
    def contar_palavras_robustamente(texto):
        if pd.isna(texto) or texto == '':
            return 0
        palavras = re.findall(r'\b\w+\b', str(texto).lower())
        return len(palavras)

    df_selecionado.loc[:, 'contagem_palavras'] = df_selecionado['transcription'].apply(contar_palavras_robustamente)
    df_filtrado_palavras = df_selecionado[df_selecionado['contagem_palavras'] <= limite_palavras_transcricao].copy()


    print(f"Linhas após filtro de contagem de palavras (<= {limite_palavras_transcricao}): {len(df_filtrado_palavras)}")

    # --- Tarefa 3: Filtrar vídeos com alta frequência da palavra "eu" ---
    def calcular_frequencia_eu(row):
        texto = row['transcription']
        contagem_total_palavras = row['contagem_palavras']

        if contagem_total_palavras == 0:
            return 0

        # Contar "eu" de forma case-insensitive, como palavra isolada
        # Usar split() para simplicidade aqui, já que contamos palavras totais de forma mais robusta antes.
        # Para maior precisão na contagem de "eu" como palavra inteira, regex \b(eu)\b seria melhor
        palavras_texto = str(texto).lower().split()
        contagem_eu = palavras_texto.count("eu")

        return contagem_eu / contagem_total_palavras

    # Aplicar a função para criar uma nova coluna com a frequência de "eu"
    df_filtrado_palavras.loc[:, 'frequencia_eu'] = df_filtrado_palavras.apply(calcular_frequencia_eu, axis=1)

    # Filtrar o DataFrame: manter apenas as linhas onde a frequência de "eu" é <= limite_frequencia_eu
    df_final_filtrado = df_filtrado_palavras[df_filtrado_palavras['frequencia_eu'] <= limite_frequencia_eu].copy()
    
    print(f"Linhas após filtro de frequência de 'eu' (<= {limite_frequencia_eu*100}%): {len(df_final_filtrado)}")

    # Remover colunas auxiliares antes de salvar
    colunas_para_remover = ['contagem_palavras', 'frequencia_eu']
    df_final_filtrado_para_salvar = df_final_filtrado.drop(columns=[col for col in colunas_para_remover if col in df_final_filtrado.columns])


    # Salvar o DataFrame resultante em um novo arquivo CSV
    df_final_filtrado_para_salvar.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"Arquivo '{output_filename}' gerado com sucesso!")
    print(f"Total de linhas no arquivo filtrado final: {len(df_final_filtrado_para_salvar)}")

except FileNotFoundError:
    print(f"Erro: O arquivo de entrada '{input_filename}' não foi encontrado.")
except KeyError as e:
    print(f"Erro de Chave: {e}. Verifique se as colunas existem no CSV.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")