# Estudo e Comparação de Modelos de Língua para Detecção de Fake News em Português 🕵️‍♂️📰

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Concluído-success)
![Institution](https://img.shields.io/badge/UNESP-IBILCE-red)

Repositório oficial do Trabalho de Conclusão de Curso (TCC) apresentado ao curso de Bacharelado em Ciência da Computação da **UNESP - São José do Rio Preto**.

**Autor:** Otávio Augusto Teixeira  
**Orientador:** Prof. Dr. Lucas Correia Ribas  
**Ano:** 2025

## 📄 Resumo do Projeto

A disseminação de notícias falsas (*fake news*) é um dos maiores desafios da era digital. Este projeto propôs um estudo comparativo extensivo de diferentes técnicas de Processamento de Linguagem Natural (PLN) para a classificação automática de notícias em português.

O estudo avaliou desde métodos estatísticos clássicos até os mais modernos **Grandes Modelos de Linguagem (LLMs)**, analisando qual representação vetorial (embedding) oferece o melhor desempenho na distinção entre notícias verdadeiras e falsas.

## 🛠️ Tecnologias e Modelos Utilizados

O projeto comparou 12 abordagens de representação de texto combinadas com 3 classificadores (SVM, Random Forest, Logistic Regression).

### Modelos de Linguagem (Embeddings)
* **Estatísticos/Baselines:** TF-IDF.
* **Estáticos:** Word2Vec, GloVe, FastText.
* **Contextuais (Transformers):** BERT (Multilingual Cased).
* **LLMs & APIs Modernas:**
    * OpenAI (`text-embedding-3-small`)
    * Google Gemini
    * SFR-Embedding-Mistral
    * Jina-Embeddings-v2
    * KALM, Serafim, E5.

### Bibliotecas Principais
* `scikit-learn`: Para classificadores e métricas.
* `transformers` (Hugging Face): Para modelos BERT e locais.
* `gensim`: Para Word2Vec e FastText.
* `pandas` & `numpy`: Manipulação de dados.
* `nltk` & `spacy`: Pré-processamento.

## 📊 Metodologia

O fluxo de trabalho (Pipeline) seguiu as seguintes etapas rigorosas:

1.  **Dataset:** Utilização do corpus **Fake.br-Corpus**, contendo 7.200 notícias (3.600 verdadeiras e 3.600 falsas), perfeitamente balanceado.
2.  **Pré-processamento:**
    * Limpeza de caracteres especiais.
    * Remoção de *stopwords* (testado com e sem).
    * Lemmatização.
3.  **Feature Extraction:** Geração de embeddings utilizando os modelos citados acima. Foram testadas combinações de entrada: *Apenas Título*, *Apenas Texto*, e *Completo (Título + Subtítulo + Texto)*.
4.  **Classificação:** Treinamento supervisionado utilizando validação cruzada.
5.  **Otimização:** Uso de *Grid Search* e *Random Search* para refinar os hiperparâmetros.

## 🏆 Resultados Principais

Os resultados demonstraram que, embora os LLMs sejam poderosos, técnicas clássicas bem ajustadas ainda são extremamente competitivas para esta tarefa específica.

Abaixo, os **Top 5 Melhores Resultados** (ordenados por F1-Score no conjunto de teste):

| Modelo de Embedding | Classificador | Acurácia | F1-Score | Detalhes |
| :--- | :--- | :--- | :--- | :--- |
| **OpenAI (3-small)** | Logistic Regression | **98.32%** | **0.9832** | Otimizado (RandomSearch) |
| **TF-IDF** | SVM | 97.98% | 0.9798 | Otimizado (GridSearch) |
| **TF-IDF** | Logistic Regression | 97.82% | 0.9782 | Otimizado (GridSearch) |
| **SFR-Mistral** | SVM | 97.27% | 0.9727 | Configuração Base |
| **BERT** | Logistic Regression | 96.72% | 0.9672 | Configuração Base |

> **Insight:** O modelo da OpenAI obteve o melhor desempenho global, mas o **TF-IDF** (uma técnica muito mais leve e rápida) ficou tecnicamente empatado, provando ser uma solução eficiente e de baixo custo computacional para detecção de fake news neste corpus.

### Análise por Combinação de Texto
A utilização do **conteúdo completo** (Título + Subtítulo + Texto) provou ser consistentemente superior ao uso isolado de apenas títulos ou apenas corpo do texto.

## 🚀 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/NOME_DO_REPO.git](https://github.com/SEU_USUARIO/NOME_DO_REPO.git)
    cd NOME_DO_REPO
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Estrutura de Arquivos:**
    * `/notebooks`: Jupyter Notebooks com os experimentos de cada modelo.
    * `/data`: Amostras do dataset (ou instruções para baixar o Fake.br-Corpus original).
    * `/results`: Arquivos CSV com os logs detalhados de todas as execuções.
    * `/src`: Scripts auxiliares de pré-processamento.

## 🔗 Referências

* *Monteiro, R. A., et al. (2018). "Fake.br-corpus: A fake news dataset in portuguese."*
* *Vaswani, A., et al. (2017). "Attention is all you need."*

---
Desenvolvido por **Otávio Augusto Teixeira** como requisito para obtenção do título de Bacharel em Ciência da Computação.