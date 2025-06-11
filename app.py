# app.py
import streamlit as st
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

# --- 1. Configurações Globais (sem alterações) ---
st.set_page_config(
    page_title="Sumarizador de Vídeos do Youtube",
    page_icon="🤖",
    layout="wide"
)

# ... (todas as suas constantes globais permanecem as mesmas) ...
BASE_MODEL_NAME = "recogna-nlp/ptt5-base-summ"
LORA_ADAPTER_PATH = "./ptt5_finetuned_lora_final"
DATA_CSV_PATH = "./datas/6.data_with_summaries.csv" 
TRANSCRIPTION_COLUMN = "transcription"
IDEAL_SUMMARY_COLUMN = "ideal_summary"
TITLE_COLUMN = "title"
VIDEO_ID_COLUMN = "video_id"
TASK_PREFIX = "resuma: "
SAFE_TOKENIZER_INPUT_MAX_LENGTH = 512
GEN_MIN_LENGTH = 20
GEN_MAX_NEW_TOKENS = 150
GEN_NUM_BEAMS = 4
GEN_NO_REPEAT_NGRAM_SIZE = 3
GEN_EARLY_STOPPING = True

# --- 2. Funções com Cache (sem alterações) ---
@st.cache_resource
def load_model_and_pipeline():
    # ... (código da função como estava) ...
    print("INICIANDO CARREGAMENTO DO MODELO (deve acontecer apenas uma vez)...")
    device_idx = 0 if torch.cuda.is_available() else -1
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
        tokenizer.model_max_length = SAFE_TOKENIZER_INPUT_MAX_LENGTH
        if hasattr(tokenizer, 'legacy'): tokenizer.legacy = False
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
        model_to_use = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        model_to_use.eval()
        print("Modelo fine-tuned com LoRA carregado.")
        summarizer_pipeline = pipeline("summarization", model=model_to_use, tokenizer=tokenizer, device=device_idx)
        print("Pipeline de sumarização pronta!")
        return summarizer_pipeline
    except Exception as e:
        print(f"Erro CRÍTICO ao carregar o modelo fine-tuned: {e}")
        return None

@st.cache_data
def load_data(path):
    # ... (código da função como estava) ...
    print(f"Carregando dados de: {path}")
    try:
        df = pd.read_csv(path)
        required_cols = [TRANSCRIPTION_COLUMN, IDEAL_SUMMARY_COLUMN, TITLE_COLUMN, VIDEO_ID_COLUMN]
        df.dropna(subset=required_cols, inplace=True)
        df.drop_duplicates(subset=[TITLE_COLUMN], keep='first', inplace=True)
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Erro ao carregar o CSV de dados: {e}")
        return None

# --- 3. Interface Gráfica e Lógica Principal ---
st.title("🤖 Sumarizador de vídeos do Youtube")
st.markdown("Selecione um título de notícia da lista abaixo. O IA irá gerar um resumo para você!")

summarizer = load_model_and_pipeline()
df_data = load_data(DATA_CSV_PATH)
#pega apenas alguns resumos dos conjuntos de validacao
video_ids_sample = ['e1t4dgJdWeM', 'mG2UwZHPxA8', 'IZ3iR8JdWeM', 'rZAgm5RhxcI', 'XOOnnGyIaGs', 'q8qLhla_DCE', 'wWpxjZOawoE']
df_data = df_data[df_data['video_id'].isin(video_ids_sample)]


if summarizer is None or df_data is None:
    st.error("Falha na inicialização. Verifique os logs do terminal para mais detalhes.")
    st.stop()

video_titles = ["-- Selecione um vídeo --"] + df_data[TITLE_COLUMN].tolist()
selected_title = st.selectbox(
    "Selecione um vídeo para resumir:",
    options=video_titles
)

if 'last_result' not in st.session_state:
    st.session_state.last_result = {}

if st.button("Gerar Resumo", type="primary") and selected_title != "-- Selecione um vídeo --":
    with st.spinner("Gerando resumo com o modelo fine-tuned..."):
        selected_row = df_data[df_data[TITLE_COLUMN] == selected_title].iloc[0]
        transcript_to_summarize = str(selected_row[TRANSCRIPTION_COLUMN])
        ideal_summary = str(selected_row[IDEAL_SUMMARY_COLUMN])
        video_id = selected_row[VIDEO_ID_COLUMN]
        
        generated_summary = "Falha ao gerar o resumo."
        try:
            input_for_model = TASK_PREFIX + transcript_to_summarize
            summary_output = summarizer(
                [input_for_model],
                min_length=GEN_MIN_LENGTH,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
                num_beams=GEN_NUM_BEAMS,
                early_stopping=GEN_EARLY_STOPPING,
                do_sample=False,
                truncation=True
            )
            generated_summary = summary_output[0]['summary_text']
            
            st.session_state.last_result = {
                "title": selected_title,
                "transcript": transcript_to_summarize,
                "ideal_summary": ideal_summary,
                "generated_summary": generated_summary,
                "video_id": video_id,
                "success": True # Flag para indicar sucesso
            }

        except Exception as e:
            st.session_state.last_result = {"error": f"Ocorreu um erro durante a sumarização: {e}"}
            
# Bloco de exibição (lê do session_state para persistir os resultados)
if st.session_state.last_result:
    result = st.session_state.last_result

    if result.get("error"):
        st.error(result["error"])

    elif result.get("success"):
        st.success(f"Processo concluído para: **{result['title']}**")

        # Construir o link do YouTube
        youtube_link = f"https://www.youtube.com/watch?v={result['video_id']}"
        st.markdown(f"**Link para o vídeo original:** [Assistir no YouTube]({youtube_link})")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Resumo Ideal")
            # MODIFICAÇÃO: Escapar o caractere '$' antes de exibir
            st.info(result['ideal_summary'].replace('$', '\\$'))
        
        with col2:
            st.subheader("Resumo Gerado pelo IA")
            # MODIFICAÇÃO: Escapar o caractere '$' antes de exibir
            st.success(result['generated_summary'].replace('$', '\\$'))

        with st.expander("Ver transcrição original pré-processada", expanded=False):
            # MODIFICAÇÃO: Escapar o caractere '$' antes de exibir
            st.text_area("", result['transcript'].replace('$', '\\$'), height=200)