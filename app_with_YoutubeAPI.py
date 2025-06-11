# app.py
import streamlit as st
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import time

# --- 1. Configurações Globais ---
st.set_page_config(
    page_title="Resumidor de Vídeos do YouTube",
    page_icon="✍️",
    layout="wide"
)

# Constantes de configuração do modelo
BASE_MODEL_NAME = "recogna-nlp/ptt5-base-summ"
# ATENÇÃO: Verifique se este caminho relativo está correto em relação à sua pasta de projeto
LORA_ADAPTER_PATH = "./ptt5_finetuned_lora_final" 
TASK_PREFIX = "resuma: "
SAFE_TOKENIZER_INPUT_MAX_LENGTH = 512

# Parâmetros de geração para o resumo
GEN_MIN_LENGTH = 20
GEN_MAX_NEW_TOKENS = 100
GEN_NUM_BEAMS = 4
GEN_NO_REPEAT_NGRAM_SIZE = 3
GEN_EARLY_STOPPING = True
COOLDOWN_SECONDS = 5

# Lista de Palavras/Frases de Preenchimento para limpeza
FILLER_PATTERNS_TO_REMOVE = [
    r'\b(e aí)\b', r'\b(né)\b', r'\b(tipo assim)\b',
    r'\b(ahn?)\b', r'\b(ah?)\b', r'\b(eh?)\b',
    r'\b(hmm)\b', r'\b(hum)\b',
    r'\[música\]', r'\[aplausos\]', r'\[risadas\]',
]

# --- 2. Funções Utilitárias ---

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
    except (NoTranscriptFound, TranscriptsDisabled):
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
@st.cache_resource
def load_model_and_pipeline():
    """Carrega o modelo base, aplica o adaptador LoRA e cria a pipeline. Executado uma vez."""
    print("INICIANDO CARREGAMENTO DO MODELO (deve acontecer apenas uma vez)...")
    device_idx = 0 if torch.cuda.is_available() else -1
    
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

# --- 4. Interface Gráfica e Lógica Principal ---
st.title("✍️ Sumarizador de Vídeos do YouTube")
st.markdown("Cole o link de um vídeo do YouTube abaixo para gerar um resumo usando um modelo T5 fine-tuned com LoRA.")

# Carregar o modelo e a pipeline (com cache)
try:
    summarizer = load_model_and_pipeline()
except Exception as e:
    st.error(f"Ocorreu um erro fatal ao carregar o modelo de sumarização: {e}")
    st.stop()

# Inicializar variáveis de estado da sessão
if 'last_click_time' not in st.session_state:
    st.session_state.last_click_time = 0.0
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = ""
if 'last_transcript' not in st.session_state:
    st.session_state.last_transcript = ""
if 'last_error' not in st.session_state:
    st.session_state.last_error = ""

# Componentes da interface
youtube_url = st.text_input("Link do vídeo do YouTube:", placeholder="https://www.youtube.com/...", key="url_input")

# Lógica do botão e cooldown
time_since_last_click = time.time() - st.session_state.last_click_time
is_in_cooldown = time_since_last_click < COOLDOWN_SECONDS

if st.button("Gerar Resumo", type="primary", disabled=is_in_cooldown):
    st.session_state.last_click_time = time.time() # Registrar tempo para o cooldown

    if not youtube_url.strip():
        st.warning("Por favor, insira uma URL do YouTube.")
        st.rerun() # Re-executa para o aviso desaparecer na próxima ação
    else:
        # Limpar resultados antigos e iniciar o processamento
        st.session_state.last_summary = ""
        st.session_state.last_error = ""
        st.session_state.last_transcript = ""
        
        with st.spinner("Processando..."):
            status_placeholder = st.empty()

            # Etapa 1: Obter Transcrição
            status_placeholder.info("1/3 - Buscando transcrição do vídeo...")
            transcript, error_msg = get_youtube_transcript_text(youtube_url)
            
            if error_msg:
                st.session_state.last_error = error_msg
            else:
                # Etapa 2: Pré-processar Transcrição
                status_placeholder.info("2/3 - Pré-processando o texto...")
                preprocessed_transcript = preprocess_transcription(transcript)
                st.session_state.last_transcript = preprocessed_transcript # Salvar para exibição
                
                if not preprocessed_transcript.strip():
                    st.session_state.last_error = "Transcrição vazia após pré-processamento."
                else:
                    # Etapa 3: Gerar Resumo
                    status_placeholder.info("3/3 - Gerando o resumo com o modelo fine-tuned...")
                    try:
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
                        st.session_state.last_summary = summary_output[0]['summary_text']
                    except Exception as e:
                        st.session_state.last_error = f"Ocorreu um erro durante a sumarização: {e}"
            
            status_placeholder.empty() # Limpa a mensagem de status "em progresso"
        
        st.rerun() # Re-executa uma vez para atualizar a tela com os resultados finais

# --- Bloco de exibição (sempre ativo, lê do session_state) ---
if st.session_state.last_error:
    st.error(st.session_state.last_error)

# Exibe a transcrição e o resumo se eles existirem no estado da sessão
if st.session_state.last_transcript:
    with st.expander("Ver transcrição pré-processada", expanded=False):
        st.text_area("", st.session_state.last_transcript, height=150, key="transcript_output_area")

if st.session_state.last_summary:
    st.success("Resumo gerado com sucesso!")
    st.text_area("Resumo:", st.session_state.last_summary, height=250, key="summary_output_area")

# Lógica do cooldown no final para não interferir na exibição do resultado
if is_in_cooldown:
    remaining_time = COOLDOWN_SECONDS - (time.time() - st.session_state.last_click_time)
    # Mostra um "brinde" e força re-execução para atualizar o estado do botão
    if remaining_time > 0:
        st.toast(f"Aguarde {remaining_time:.1f} segundos...")
        time.sleep(1) # Pequena pausa para o usuário ver o toast
        st.rerun()