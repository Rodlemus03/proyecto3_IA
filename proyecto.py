import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# --- Preprocesamiento y entrenamiento ---
df = pd.read_csv('spam_ham.csv', sep=';', encoding='cp1252')
df['Label'] = df['Label'].str.strip().str.lower()
df = df[df['Label'].isin(['spam', 'ham'])]
df = df[df['SMS_TEXT'].notna() & df['SMS_TEXT'].str.strip().ne('')]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    if re.search(r'http[s]?://|www\.', text):
        tokens.append("url_detected")
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and w not in string.punctuation and not w.isdigit()]
    return tokens

df['tokens'] = df['SMS_TEXT'].apply(preprocess)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

# Inyectar token url_detected en ejemplos SPAM
train_df.loc[train_df['Label'] == 'spam', 'tokens'] = train_df[train_df['Label'] == 'spam']['tokens'].apply(lambda t: t + ['url_detected'])

def word_probs(df_subset):
    word_counter = Counter()
    for tokens in df_subset['tokens']:
        word_counter.update(tokens)
    total = sum(word_counter.values())
    return {w: count / total for w, count in word_counter.items()}

spam_df = train_df[train_df['Label'] == 'spam']
ham_df = train_df[train_df['Label'] == 'ham']
P_spam = len(spam_df) / len(train_df)
P_ham = 1 - P_spam
P_w_spam = word_probs(spam_df)
P_w_ham = word_probs(ham_df)

def combined_prob(tokens):
    probs = []
    for w in tokens:
        pw_s = P_w_spam.get(w, 1e-6)
        pw_h = P_w_ham.get(w, 1e-6)
        p = (pw_s * P_spam) / ((pw_s * P_spam) + (pw_h * P_ham))
        probs.append(p)
    if not probs:
        return 0
    num = np.prod(probs)
    denom = num + np.prod([1 - p for p in probs])
    return num / denom if denom != 0 else 0

def evaluar_mensaje(texto):
    tokens = preprocess(texto)
    score = combined_prob(tokens)
    top_palabras = sorted([(w, (P_w_spam.get(w, 1e-6) * P_spam) / ((P_w_spam.get(w, 1e-6) * P_spam) + (P_w_ham.get(w, 1e-6) * P_ham))) for w in tokens], key=lambda x: -x[1])
    return score, top_palabras[:3]

# --- Interfaz Gráfica ---
root = tk.Tk()
root.title("Clasificador SPAM/HAM")
root.geometry("600x400")
root.configure(bg='#f0f0f0')

frame = ttk.Frame(root, padding=20)
frame.pack(expand=True, fill="both")

label = ttk.Label(frame, text="Ingrese el mensaje de texto:", font=('Segoe UI', 12))
label.pack(anchor="w")

text_input = ScrolledText(frame, height=6, font=('Segoe UI', 11))
text_input.pack(fill="both", pady=10)

result_label = ttk.Label(frame, text="", font=('Segoe UI', 11, 'bold'))
result_label.pack(pady=5)

def analizar():
    texto = text_input.get("1.0", tk.END).strip()
    if not texto:
        messagebox.showwarning("Atención", "Por favor ingrese un texto para analizar.")
        return
    score, top3 = evaluar_mensaje(texto)
    clase = "SPAM" if score > 0.5 else "HAM"
    palabras = ', '.join([f"{w} ({p:.2f})" for w, p in top3])
    result_label.config(text=f"Resultado: {clase}\nProbabilidad de SPAM: {score:.2f}\nTop 3 palabras predictivas: {palabras}")

analyze_button = ttk.Button(frame, text="Analizar", command=analizar)
analyze_button.pack(pady=10)

root.mainloop()