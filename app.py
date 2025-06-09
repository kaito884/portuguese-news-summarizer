# app.py
import streamlit as st
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GenerationConfig
from peft import PeftModel
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# --- 1. Configurações Globais ---
st.set_page_config(
    page_title="Resumidor de Vídeos do YouTube",
    page_icon="✍️",
    layout="wide"
)

# --- Configurações do Modelo e Geração ---
BASE_MODEL_NAME = "recogna-nlp/ptt5-base-summ"
LORA_ADAPTER_PATH = "./ptt5_finetuned_lora_final"
TOKEN_AUTH = None 
TASK_PREFIX = "resuma: "
SAFE_TOKENIZER_INPUT_MAX_LENGTH = 512

GEN_MIN_LENGTH = 20
GEN_MAX_NEW_TOKENS = 100
GEN_NUM_BEAMS = 4
GEN_NO_REPEAT_NGRAM_SIZE = 3
GEN_EARLY_STOPPING = True

# --- 2. Funções Utilitárias ---
# Lista de Palavras/Frases de Preenchimento
FILLER_PATTERNS_TO_REMOVE = [
    r'\b(e aí)\b', r'\b(né)\b', r'\b(tipo assim)\b',
    r'\b(ahn?)\b', r'\b(ah?)\b', r'\b(eh?)\b',
    r'\b(hmm)\b', r'\b(hum)\b',
    r'\[música\]', r'\[aplausos\]', r'\[risadas\]',
]

def get_youtube_transcript_text(youtube_url):
    """Extrai a transcrição de uma URL do YouTube."""
    video_id = None
    patterns = [r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", r"youtu\.be\/([0-9A-Za-z_-]{11})", r"embed\/([0-9A-Za-z_-]{11})", r"shorts\/([0-9A-Za-z_-]{11})"]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break
    if not video_id:
        return None, "URL do YouTube inválida ou formato não reconhecido."
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'pt-BR', 'en'])
        full_transcript = " ".join([item['text'] for item in transcript_list])
        return full_transcript, None
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        return None, f"Não foi possível obter a transcrição. Pode não existir em PT/EN ou estar desabilitada. (ID: {video_id})"
    except Exception as e:
        return None, f"Erro inesperado ao obter transcrição: {str(e)}"

def preprocess_transcription(text):
    """Aplica pré-processamento ao texto da transcrição."""
    if not isinstance(text, str): return ""
    processed_text = text.lower()
    for pattern in FILLER_PATTERNS_TO_REMOVE:
        processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

# --- 3. Carregamento do Modelo com Cache do Streamlit ---
# @st.cache_resource diz ao Streamlit para carregar o modelo apenas uma vez.
@st.cache_resource
def load_model_and_pipeline():
    """Carrega o modelo base, aplica o adaptador LoRA e cria a pipeline."""
    print("INICIANDO CARREGAMENTO DO MODELO (deve acontecer apenas uma vez)...")
    device_idx = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    tokenizer.model_max_length = SAFE_TOKENIZER_INPUT_MAX_LENGTH
    if hasattr(tokenizer, 'legacy'): tokenizer.legacy = False
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    
    model_to_use = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model_to_use.eval()
    print("Modelo fine-tuned com LoRA carregado.")

    summarizer_pipeline = pipeline(
        "summarization",
        model=model_to_use,
        tokenizer=tokenizer,
        device=device_idx
    )
    print("Pipeline de sumarização pronta!")
    return summarizer_pipeline

# --- 4. Interface Gráfica da Aplicação Web com Streamlit ---
st.title("✍️ Sumarizador de Vídeos do YouTube")
st.markdown("Cole o link de um vídeo do YouTube abaixo para gerar um resumo usando um modelo T5 fine-tuned com LoRA.")

# Carregar o modelo e a pipeline usando o cache
try:
    summarizer = load_model_and_pipeline()
except Exception as e:
    st.error(f"Ocorreu um erro fatal ao carregar o modelo de sumarização: {e}")
    st.stop() # Interrompe a execução do app se o modelo não puder ser carregado

# Componentes da interface
youtube_url = st.text_input("Link do vídeo do YouTube:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Gerar Resumo", type="primary"):
    if not youtube_url.strip():
        st.warning("Por favor, insira uma URL do YouTube.")
    else:
        with st.spinner("Processando... Por favor, aguarde."):
            # Etapa 1: Obter Transcrição
            st.info("Buscando transcrição do vídeo...")
            transcript, error = get_youtube_transcript_text(youtube_url)
            
            if error:
                st.error(error)
            else:
                # Etapa 2: Pré-processar Transcrição
                st.info("Pré-processando o texto...")
                preprocessed_transcript = preprocess_transcription(transcript)
                
                # Exibir a transcrição (opcional, mas útil para o usuário)
                with st.expander("Ver transcrição pré-processada"):
                    st.text_area("", preprocessed_transcript, height=150)

                # Etapa 3: Gerar Resumo
                st.info("Gerando o resumo com o modelo fine-tuned...")
                try:
                    # Os parâmetros de geração serão usados pela pipeline
                    summary_output = summarizer(
                        TASK_PREFIX + preprocessed_transcript,
                        min_length=GEN_MIN_LENGTH,
                        max_new_tokens=GEN_MAX_NEW_TOKENS,
                        no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
                        num_beams=GEN_NUM_BEAMS,
                        early_stopping=GEN_EARLY_STOPPING,
                        do_sample=False,
                        truncation=True
                    )
                    
                    # Etapa 4: Exibir Resultado
                    st.success("Resumo gerado com sucesso!")
                    st.text_area("Resumo:", summary_output[0]['summary_text'], height=200)

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a sumarização: {e}")