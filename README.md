# Sumarizador de Notícias em Português com Fine-tuning de Modelos T5

## Descrição

Este projeto implementa um pipeline para sumarização de transcrições de vídeos de notícias em português do YouTube. O sistema inclui scripts para obtenção de transcrições, filtragem e pré-processamento de dados, fine-tuning de um modelo de linguagem T5 (especificamente o `recogna-nlp/ptt5-base-summ`) usando LoRA (Low-Rank Adaptation), e uma interface gráfica (GUI) para sumarizar vídeos a partir de uma URL do YouTube usando o modelo fine-tuned.

## Funcionalidades Principais

* Obtenção automática de transcrições de vídeos do YouTube.
* Filtragem de vídeos baseada em contagem de palavras
* Pré-processamento de texto para limpeza das transcrições.
* Fine-tuning de um modelo T5 para sumarização abstrativa em português usando LoRA.
* Script para avaliação do modelo fine-tuned em comparação com o modelo base e resumos ideais.
* Geração de resumos em lote a partir de um arquivo CSV de transcrições.
* Interface gráfica (GUI) para inserir uma URL do YouTube e obter um resumo gerado pelo modelo fine-tuned.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

PLN/

├── datas/                             # Pasta para arquivos CSV de dados

│   ├── 1.youtube_data.csv             # (Opcional, dados brutos iniciais, pode ser grande)

│   ├── 2.data_with_transcription.csv  # Saída do script 1

│   ├── 3.filtered_data.csv            # Saída do script 2

│   ├── 4.data_preprocessed.csv        # Saída do script 3

│   ├── 4.train_ideal_summary.csv      # Dados de treino para fine-tuning (transcrição, ideal_summary)

│   ├── 4.validation_ideal_summary.csv # Dados de validação para fine-tuning

│   ├── 6.evaluation_comparison_results.csv # Saída do script de validação do fine-tuning

│   └── 7.data_with_summaries.csv      # Saída do script de geração em lote

├── frontend/                          # Pasta para a interface gráfica

│   ├── gui.py

│   └── main.py

├── ptt5_finetuned_lora_final/         # Pasta com o adaptador LoRA treinado

│   ├── adapter_config.json

│   ├── adapter_model.bin

│   └── ...

├── 1.get_transcription.py             # Script para obter transcrições

├── 2.filter_videos.py                 # Script para filtrar os dados

├── 3.pre_process.py                   # Script para pré-processar as transcrições

├── 4.fine_tunning.py                  # Script para o fine-tuning do modelo LoRA

├── 5.validate_fine_tunning.py         # Script para avaliar o modelo fine-tuned (comparação)

├── 6.generate_summary.py              # Script para gerar resumos em lote com o modelo fine-tuned

├── requirements.txt                   # Dependências do Python

└── README.md                          # Este arquivo

## Pré-requisitos

* Python 3.10 ou superior
* Git
* (Opcional, mas recomendado para fine-tuning e inferência rápida) GPU NVIDIA com CUDA configurado.

## Configuração do Ambiente e Instalação

1.  **Clone o Repositório (se estiver no GitHub):**
    ```bash
    git clone [https://github.com/seu_usuario/seu_projeto.git](https://github.com/seu_usuario/seu_projeto.git)
    cd seu_projeto
    ```

2.  **Obtenha o Dataset Inicial:**
    Este projeto utiliza inicialmente um dataset de vídeos em alta do YouTube para extrair transcrições. O arquivo `1.youtube_data.csv` (ou um arquivo base similar de onde os `video_id`s são extraídos) não está incluído no repositório devido ao seu tamanho.
    * **Faça o download do dataset de vídeos em alta do Brasil em:** [YouTube Trending Video Dataset (Brazil)](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset?select=BR_youtube_trending_data.csv)
    * Após o download, renomeie o arquivo `BR_youtube_trending_data.csv` para `1.youtube_data.csv` (ou o nome que o script `1.get_transcription.py` espera como entrada).
    * Coloque este arquivo CSV baixado e renomeado na pasta `datas/` do seu projeto.


3.  **Crie e Ative um Ambiente Virtual:**
    (Recomendado para isolar as dependências do projeto)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # OU
    # venv\Scripts\activate    # Windows
    ```

4.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    Certifique-se de que o `requirements.txt` contém todas as bibliotecas necessárias, como: `pandas`, `torch`, `transformers`, `accelerate`, `peft`, `bitsandbytes`, `sentencepiece`, `youtube_transcript_api`, `nltk`, `evaluate`, `rouge_score`.


## Modelo de Sumarização

* **Modelo Base:** O fine-tuning é realizado sobre o modelo `recogna-nlp/ptt5-base-summ`.
* **Adaptador LoRA:** O script `4.fine_tunning.py` treina um adaptador LoRA. Os pesos e a configuração deste adaptador são salvos no diretório especificado por `OUTPUT_DIR` nesse script (ex: `./ptt5_finetuned_lora_final/`).
* **Caminho do Adaptador:** Para usar o modelo fine-tuned nos scripts `5.validate_fine_tunning.py`, `6.generate_summary.py` e `gui.py`, certifique-se de que a constante `LORA_ADAPTER_PATH` nesses scripts aponte para o diretório correto onde o adaptador LoRA foi salvo.

## Scripts e Como Usar

A ordem recomendada para executar os scripts de processamento de dados e treinamento é:

1.  **`1.get_transcription.py`**
    * **Propósito:** Busca transcrições de vídeos do YouTube a partir de um CSV inicial (`datas/1.youtube_data.csv`) que contém `video_id`s.
    * **Saída:** Gera `datas/2.data_with_transcription.csv` com as transcrições adicionadas.
    * **Como usar:** `python 1.get_transcription.py` (ajuste caminhos e parâmetros dentro do script se necessário).

2.  **`2.filter_videos.py`**
    * **Propósito:** Filtra o arquivo CSV com transcrições com base em critérios como contagem máxima de palavras e frequência da palavra "eu".
    * **Entrada:** `datas/2.data_with_transcription.csv`
    * **Saída:** `datas/3.filtered_data.csv`
    * **Como usar:** `python 2.filter_videos.py`

3.  **`3.pre_process.py`**
    * **Propósito:** Aplica limpeza e pré-processamento às transcrições filtradas (converte para minúsculas, remove fillers, etc.).
    * **Entrada:** `datas/3.filtered_data.csv`
    * **Saída:** `datas/4.data_preprocessed.csv`
    * **Como usar:** `python 3.pre_process.py`

4.  **`4.fine_tunning.py`**
    * **Propósito:** Realiza o fine-tuning do modelo T5 usando LoRA.
    * **Preparação dos Dados:** Requer dois arquivos CSV:
        * `TRAIN_CSV_FILE_PATH` (ex: `datas/4.train_ideal_summary.csv`): Contendo colunas `transcription` (pré-processada) e `ideal_summary` para treino.
        * `VALIDATION_CSV_FILE_PATH` (ex: `datas/4.validation_ideal_summary.csv`): Com a mesma estrutura para validação.
    * **Saída:** Salva o adaptador LoRA treinado no diretório `OUTPUT_DIR` (ex: `./ptt5_finetuned_lora_final/`).
    * **Como usar:** `python 4.fine_tunning.py` (ajuste caminhos e hiperparâmetros dentro do script).

5.  **`extra.validate_fine_tunning.py`**
    * **Propósito:** Gera resumos usando o modelo base e o modelo fine-tuned com LoRA para um conjunto de validação, permitindo a comparação lado a lado com os resumos ideais.
    * **Entrada:** `VALIDATION_CSV_PATH` (ex: `datas/4.validation_ideal_summary.csv`), modelo base, e o adaptador LoRA salvo.
    * **Saída:** Um arquivo CSV de comparação (ex: `datas/6.evaluation_comparison_results.csv`).
    * **Como usar:** `python 5.validate_fine_tunning.py`

6.  **`5.generate_summary.py`**
    * **Propósito:** Gera resumos em lote para um arquivo CSV de transcrições pré-processadas usando o modelo fine-tuned.
    * **Entrada:** `CSV_FILE_PATH` (ex: `datas/4.data_preprocessed.csv`) e o adaptador LoRA.
    * **Saída:** `OUTPUT_CSV_FILE_PATH` (ex: `datas/7.data_with_summaries.csv`) com uma nova coluna de resumos.
    * **Como usar:** `python 6.generate_summary.py`

7.  **Interface Gráfica (GUI)**
    * **Propósito:** Permite ao usuário inserir uma URL de vídeo do YouTube e obter um resumo gerado pelo modelo fine-tuned.
    * **Arquivos:** `frontend/main.py` e `frontend/gui.py`.
    * **Como usar:**
        ```bash
        cd frontend
        python main.py
        ```
        Ou, se estiver na raiz do projeto: `python frontend/main.py`
    * A GUI carregará o modelo fine-tuned (o caminho do adaptador LoRA está configurado em `gui.py`).
