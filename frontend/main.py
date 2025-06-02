# Bibliotecas a serem instaladas: (acoh que e so isso, mas se precisar de mais bibliotecas, instale-as)
# pip install pandas torch transformers accelerate peft sentencepiece youtube_transcript_api Pillow

# main.py
import tkinter as tk
from gui import SummarizerGUI 

if __name__ == "__main__":
    root = tk.Tk()
    app = SummarizerGUI(root)
    root.mainloop()