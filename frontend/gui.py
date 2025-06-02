import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import re
import pandas as pd # Para pd.isna em preprocess_transcription_for_gui

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# --- Importações para o Modelo de Sumarização ---
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GenerationConfig
from peft import PeftModel

# --- Configurações Globais para o Modelo e Sumarização ---
BASE_MODEL_NAME = "recogna-nlp/ptt5-base-summ"
LORA_ADAPTER_PATH = "../ptt5_finetuned_lora_final" # Caminho para o adaptador LoRA treinado
TOKEN_AUTH = None 
TASK_PREFIX = "resuma: "
SAFE_TOKENIZER_INPUT_MAX_LENGTH = 512 # Limite de entrada do tokenizador para PTT5-base

# Parâmetros de geração para o resumo
GEN_MIN_LENGTH = 20
GEN_MAX_NEW_TOKENS = 100
GEN_NUM_BEAMS = 4
GEN_NO_REPEAT_NGRAM_SIZE = 3
GEN_EARLY_STOPPING = True

# Variáveis globais para o modelo e pipeline (carregados uma vez)
g_summarizer_pipeline = None
g_model_tokenizer_loaded = False
g_device_name = "cuda" if torch.cuda.is_available() else "cpu"
g_device_idx = 0 if g_device_name == "cuda" else -1

# Lista de Palavras/Frases de Preenchimento para pré-processamento
FILLER_PATTERNS_TO_REMOVE = [
    r'\b(e aí)\b', r'\b(né)\b', r'\b(tipo assim)\b',
    r'\b(ahn?)\b', r'\b(ah?)\b', r'\b(eh?)\b',
    r'\b(hmm)\b', r'\b(hum)\b',
    r'\[música\]', r'\[aplausos\]', r'\[risadas\]',
    # Adicione outros padrões conforme necessário
    # r'\b(então)\b', # Use com cautela, pode ser semanticamente importante
    # r'\b(assim)\b',  # Use com cautela
]

# --- Funções Utilitárias ---

def get_youtube_transcript_text(youtube_url):
    """
    Extrai o texto da transcrição de uma URL do YouTube.
    Retorna (texto_da_transcrição, None) em sucesso, ou (None, mensagem_de_erro) em falha.
    """
    video_id = None
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})", 
        r"embed\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break
    
    if not video_id:
        return None, "URL do YouTube inválida ou formato não reconhecido."

    try:
        # Tenta primeiro pt-BR, depois pt, depois en como fallback
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt-BR', 'pt', 'en'])
        full_transcript = " ".join([item['text'] for item in transcript_list])
        if not full_transcript.strip():
            return None, "Transcrição obtida está vazia."
        return full_transcript, None
    except NoTranscriptFound:
        return None, f"Nenhuma transcrição em PT/EN encontrada para o vídeo ID: {video_id}."
    except TranscriptsDisabled:
        return None, f"Transcrição desabilitada para o vídeo ID: {video_id}."
    except Exception as e:
        return None, f"Erro ao obter transcrição: {str(e)}"

def preprocess_transcription_for_gui(text):
    """
    Aplica etapas de pré-processamento a um texto de transcrição.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    processed_text = text.lower()
    for pattern in FILLER_PATTERNS_TO_REMOVE:
        processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    # Remove específicos no final, se ainda existirem após a limpeza geral
    if processed_text.endswith(" e aí"):
      processed_text = processed_text[:-4].strip()
    if processed_text.endswith(" aí"): 
      processed_text = processed_text[:-3].strip()
    return processed_text

def load_fine_tuned_model_and_pipeline():
    """
    Carrega o modelo base, aplica o adaptador LoRA e cria uma pipeline de sumarização.
    Deve ser chamado apenas uma vez para otimizar o desempenho.
    """
    global g_summarizer_pipeline, g_model_tokenizer_loaded, g_device_idx, g_device_name

    if g_model_tokenizer_loaded:
        print("Modelo e pipeline já carregados.")
        return True

    print(f"Carregando modelo base '{BASE_MODEL_NAME}' e aplicando adaptador LoRA de '{LORA_ADAPTER_PATH}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=TOKEN_AUTH, use_fast=True)
        
        # Ajustar tokenizer.model_max_length para o limite do modelo base
        tokenizer.model_max_length = SAFE_TOKENIZER_INPUT_MAX_LENGTH
        print(f"  > tokenizer.model_max_length para pipeline ajustado para: {tokenizer.model_max_length}")

        if hasattr(tokenizer, 'legacy'): 
            tokenizer.legacy = False # Para suprimir warning do T5Tokenizer

        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME, token=TOKEN_AUTH)
        
        model_to_use = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        model_to_use.eval() # Colocar em modo de avaliação
        print("Modelo fine-tuned com LoRA carregado.")

        # Definir configuração de geração para o modelo (influencia a pipeline)
        if model_to_use.generation_config is None:
            model_to_use.generation_config = GenerationConfig()
        
        model_to_use.generation_config.min_length = GEN_MIN_LENGTH
        model_to_use.generation_config.max_new_tokens = GEN_MAX_NEW_TOKENS
        model_to_use.generation_config.num_beams = GEN_NUM_BEAMS
        model_to_use.generation_config.early_stopping = GEN_EARLY_STOPPING
        model_to_use.generation_config.no_repeat_ngram_size = GEN_NO_REPEAT_NGRAM_SIZE
        model_to_use.generation_config.do_sample = False
        if tokenizer.pad_token_id is not None: # Necessário para modelos T5
             model_to_use.generation_config.decoder_start_token_id = tokenizer.pad_token_id

        g_summarizer_pipeline = pipeline(
            "summarization",
            model=model_to_use,
            tokenizer=tokenizer,
            device=g_device_idx # -1 para CPU, 0 para primeira GPU
        )
        g_model_tokenizer_loaded = True
        print("Pipeline de sumarização com modelo fine-tuned pronta!")
        return True
    except Exception as e:
        print(f"Erro CRÍTICO ao carregar o modelo fine-tuned: {e}")
        g_summarizer_pipeline = None 
        g_model_tokenizer_loaded = False
        import traceback
        traceback.print_exc()
        return False

class SummarizerGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.geometry("1000x700")
        self.root.title("Resumidor de Vídeos do YouTube (PLN Fine-Tuned)")

        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        titulo_label = ttk.Label(main_frame, text="Resumidor de Vídeos do YouTube", font=("Arial", 18, "bold"))
        titulo_label.pack(pady=10)

        input_frame = ttk.Frame(main_frame, padding="5")
        input_frame.pack(pady=5, fill=tk.X)

        input_text_label = ttk.Label(input_frame, text="Link do YouTube:")
        input_text_label.pack(side=tk.LEFT, padx=(0,5))
        self.input_box_var = tk.StringVar()
        self.input_box = ttk.Entry(input_frame, textvariable=self.input_box_var, font=('Arial', 11), width=70)
        self.input_box.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.enter_btn = ttk.Button(input_frame, text="Gerar Resumo", command=self.start_processing_thread)
        self.enter_btn.pack(side=tk.LEFT, padx=(5,0))
        
        self.status_label_var = tk.StringVar()
        self.status_label_var.set("Aguardando URL...") 
        status_label = ttk.Label(main_frame, textvariable=self.status_label_var, font=("Arial", 10), anchor="w")
        status_label.pack(pady=5, fill=tk.X)

        output_frame = ttk.LabelFrame(main_frame, text="Resumo Gerado", padding="5")
        output_frame.pack(pady=10, padx=5, fill=tk.BOTH, expand=True)
        
        self.output_text_area = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=80, height=15, font=("Arial", 10))
        self.output_text_area.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        self.output_text_area.config(state=tk.DISABLED)

        # Carregar o modelo ao iniciar a GUI (pode ser em thread para não travar)
        self._initialize_model_async()

    def _initialize_model_async(self):
        if not g_model_tokenizer_loaded:
            self.status_label_var.set("Carregando modelo de sumarização (pode levar um momento)...")
            self.root.update_idletasks()
            
            # Rodar o carregamento do modelo em uma thread para não congelar a GUI
            model_load_thread = threading.Thread(target=self._load_model_worker)
            model_load_thread.daemon = True
            model_load_thread.start()
        else:
            self.status_label_var.set("Modelo já carregado. Pronto.")
            
    def _load_model_worker(self):
        if not load_fine_tuned_model_and_pipeline():
            self.root.after(0, lambda: messagebox.showerror("Erro de Inicialização", 
                            "Não foi possível carregar o modelo de sumarização. Verifique os logs do console."))
            self.status_label_var.set("Falha ao carregar modelo. Reinicie o programa.")
        else:
            self.status_label_var.set("Modelo carregado. Pronto para resumir!")
        self.root.update_idletasks()


    def _update_gui_status(self, status_msg, output_msg=None, enable_button=None):
        """Atualiza a GUI na thread principal."""
        self.status_label_var.set(status_msg)
        if output_msg is not None:
            self.output_text_area.config(state=tk.NORMAL)
            self.output_text_area.delete(1.0, tk.END)
            self.output_text_area.insert(tk.INSERT, output_msg)
            self.output_text_area.config(state=tk.DISABLED)
        if enable_button is not None:
            self.enter_btn.config(state=tk.NORMAL if enable_button else tk.DISABLED)
        self.root.update_idletasks()

    def _processing_task(self, youtube_url):
        global g_summarizer_pipeline, g_model_tokenizer_loaded

        if not g_model_tokenizer_loaded or g_summarizer_pipeline is None:
            self.root.after(0, self._update_gui_status, "Erro: Modelo de sumarização não está carregado.", "Falha crítica.", True)
            return

        self.root.after(0, self._update_gui_status, "Obtendo transcrição do YouTube...")
        transcript, error_msg = get_youtube_transcript_text(youtube_url)

        if error_msg:
            self.root.after(0, self._update_gui_status, f"Erro: {error_msg}", "", True)
            return
        if not transcript:
            self.root.after(0, self._update_gui_status, "Erro: Transcrição não pôde ser obtida ou está vazia.", "", True)
            return

        self.root.after(0, self._update_gui_status, "Pré-processando transcrição...")
        preprocessed_transcript = preprocess_transcription_for_gui(transcript)

        if not preprocessed_transcript.strip():
            self.root.after(0, self._update_gui_status, "Erro: Transcrição vazia após pré-processamento.", "", True)
            return
        
        word_count = len(preprocessed_transcript.split())
        min_words_for_summary = 20 
        if word_count < min_words_for_summary:
            self.root.after(0, self._update_gui_status, 
                            f"Aviso: Transcrição muito curta ({word_count} palavras). O resumo pode não ser ideal.", 
                            preprocessed_transcript + "\n\n(Transcrição original muito curta)")
            # Continuar mesmo assim, mas o usuário está avisado.

        self.root.after(0, self._update_gui_status, "Gerando resumo...")
        generated_summary = "Falha ao gerar resumo." # Default
        try:
            input_for_model = TASK_PREFIX + preprocessed_transcript
            
            # A pipeline usa o generation_config do modelo, que já foi ajustado
            summary_output_list = g_summarizer_pipeline(
                [input_for_model],
                # Parâmetros aqui sobrescreveriam o model.generation_config para esta chamada específica.
                # Se o model.generation_config está bem definido, não precisa repeti-los aqui.
                # Mas para garantir, podemos mantê-los:
                min_length=GEN_MIN_LENGTH,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                num_beams=GEN_NUM_BEAMS,
                early_stopping=GEN_EARLY_STOPPING,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
                do_sample=False,
                truncation=True 
            )
            if summary_output_list and isinstance(summary_output_list, list) and 'summary_text' in summary_output_list[0]:
                generated_summary = summary_output_list[0]['summary_text']
                self.root.after(0, self._update_gui_status, "Resumo gerado com sucesso!", generated_summary, True)
            else:
                raise ValueError("Formato de saída da pipeline inesperado.")

        except Exception as e:
            self.root.after(0, self._update_gui_status, f"Erro durante a sumarização: {str(e)}", f"Falha ao gerar resumo.\nDetalhes: {str(e)}", True)
            import traceback
            traceback.print_exc() # No console para depuração
        # finally: # O botão é reabilitado dentro do _update_gui_status quando apropriado

    def start_processing_thread(self):
        youtube_url = self.input_box_var.get()
        if not youtube_url.strip():
            messagebox.showwarning("Entrada Vazia", "Por favor, insira uma URL do YouTube.")
            return

        # Desabilitar botão e limpar/definir status inicial via _update_gui_status
        self._update_gui_status("Iniciando processamento...", "", False) # output_msg vazio, enable_button False
        
        thread = threading.Thread(target=self._processing_task, args=(youtube_url,))
        thread.daemon = True 
        thread.start()