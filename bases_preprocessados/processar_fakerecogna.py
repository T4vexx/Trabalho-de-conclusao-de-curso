import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
import nltk
import string

# Baixar recursos necessários do NLTK (apenas para stop words)
print("Baixando recursos do NLTK...")
nltk.download('stopwords', quiet=True)

# Carregar modelo de português do spaCy
print("Carregando modelo do spaCy para português...")
print("ATENÇÃO: Se o modelo não estiver instalado, execute:")
print("  python -m spacy download pt_core_news_sm")
print()

try:
    nlp = spacy.load('pt_core_news_sm')
except:
    print("ERRO: Modelo não encontrado!")
    print("Instale com: python -m spacy download pt_core_news_sm")
    print("Ou use o modelo maior e mais preciso: python -m spacy download pt_core_news_lg")
    exit(1)

# Carregar stop words do português
stop_words = set(stopwords.words('portuguese'))

def preprocess_text_with_stopwords(text):
    """
    Remove stop words e aplica lemmatização REAL usando spaCy
    """
    if pd.isna(text):
        return ""

    # Processar o texto com spaCy
    doc = nlp(text.lower())

    # Remover stop words e pontuação, manter apenas lemmas
    tokens = [token.lemma_ for token in doc 
              if token.text not in stop_words 
              and token.text not in string.punctuation
              and not token.is_space]

    return ' '.join(tokens)

def preprocess_text_only_special_chars(text):
    """
    Remove apenas caracteres especiais (., /, !, —, ,) e aplica lemmatização
    """
    if pd.isna(text):
        return ""

    # Converter para minúsculas
    text = text.lower()

    # Remover apenas os caracteres especiais especificados
    text = re.sub(r'[./!—,]', ' ', text)

    # Processar o texto com spaCy
    doc = nlp(text)

    # Aplicar lemmatização, removendo apenas espaços extras
    tokens = [token.lemma_ for token in doc if not token.is_space]

    return ' '.join(tokens)

def process_dataset(input_file, output_file1, output_file2):
    """
    Processa o dataset e gera dois arquivos Excel
    """
    print(f"Carregando dataset: {input_file}")

    # Carregar o dataset
    df = pd.read_excel(input_file)

    print(f"Dataset carregado com {len(df)} linhas e {len(df.columns)} colunas")
    print(f"Colunas: {list(df.columns)}")

    # Identificar colunas de texto para processar
    text_columns = ['Titulo', 'Subtitulo', 'Noticia']

    # Dataset 1: Com remoção de stop words e lemmatização
    print("\nProcessando Dataset 1: Remoção de stop words + lemmatização...")
    df1 = df.copy()

    for col in text_columns:
        if col in df1.columns:
            print(f"  Processando coluna: {col}")
            df1[f'{col}_processed'] = df1[col].apply(preprocess_text_with_stopwords)

    # Salvar Dataset 1
    print(f"Salvando Dataset 1 em: {output_file1}")
    df1.to_excel(output_file1, index=False, engine='openpyxl')
    print(f"✓ Dataset 1 salvo com sucesso!")

    # Dataset 2: Removendo apenas caracteres especiais e lemmatizada
    print("\nProcessando Dataset 2: Remoção de caracteres especiais (. / ! — ,) + lemmatização...")
    df2 = df.copy()

    for col in text_columns:
        if col in df2.columns:
            print(f"  Processando coluna: {col}")
            df2[f'{col}_processed'] = df2[col].apply(preprocess_text_only_special_chars)

    # Salvar Dataset 2
    print(f"Salvando Dataset 2 em: {output_file2}")
    df2.to_excel(output_file2, index=False, engine='openpyxl')
    print(f"✓ Dataset 2 salvo com sucesso!")

    print("\n" + "="*50)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*50)
    print(f"Arquivo 1: {output_file1}")
    print(f"Arquivo 2: {output_file2}")

    # Mostrar exemplo de processamento
    print("\n" + "="*50)
    print("EXEMPLO DE PROCESSAMENTO:")
    print("="*50)

    # Exemplo com a primeira coluna de texto disponível
    for col in text_columns:
        if col in df.columns and len(df) > 0:
            idx = 0
            print(f"\n{col} original:")
            print(f"  {df[col].iloc[idx]}")
            print(f"\n{col} processado (Dataset 1 - sem stop words):")
            print(f"  {df1[f'{col}_processed'].iloc[idx]}")
            print(f"\n{col} processado (Dataset 2 - sem . / ! — ,):")
            print(f"  {df2[f'{col}_processed'].iloc[idx]}")
            break

if __name__ == "__main__":
    # Configurar os nomes dos arquivos
    INPUT_FILE = "fakerecogna.xlsx"  # Nome do arquivo de entrada
    OUTPUT_FILE1 = "dataset_sem_stopwords_lemmatizado.xlsx"
    OUTPUT_FILE2 = "dataset_sem_caracteres_especiais_lemmatizado.xlsx"

    # Processar o dataset
    process_dataset(INPUT_FILE, OUTPUT_FILE1, OUTPUT_FILE2)
