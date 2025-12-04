import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configurações
DATASETS = [
    '../../bases_preprocessados/dataset_sem_stopwords_lemmatizado.xlsx',
    '../../bases_preprocessados/dataset_sem_caracteres_especiais_lemmatizado.xlsx'
]

FIELD_COMBINATIONS = {
    'titulo': ['Titulo'],
    'texto': ['Noticia'],
    'subtitulo': ['Subtitulo'],
    'titulo_subtitulo': ['Titulo', 'Subtitulo'],
    'completo': ['Titulo', 'Subtitulo', 'Noticia']
}

CLASSIFIERS = {
    'SVM': SVC(kernel='linear', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
}

# Diretórios para checkpoints
CHECKPOINT_DIR = 'checkpoints_tfidf'
RESULTS_DIR = 'results_tfidf'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class CheckpointManager:
    """Gerenciador de checkpoints para salvar progresso"""

    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.completed = self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return set()

    def save_checkpoint(self):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.completed, f)

    def is_completed(self, key):
        return key in self.completed

    def mark_completed(self, key):
        self.completed.add(key)
        self.save_checkpoint()

def log_print(message, log_file):
    """Imprime na tela e salva no arquivo de log"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def combine_fields(df, fields, use_processed=True):
    """Combina campos de texto"""
    suffix = '_processed' if use_processed else ''
    texts = []

    for _, row in df.iterrows():
        combined = []
        for field in fields:
            field_name = field + suffix
            if field_name in df.columns and pd.notna(row[field_name]):
                combined.append(str(row[field_name]))
        texts.append(' '.join(combined))

    return texts

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier_name, classifier, log_file):
    """Treina e avalia um classificador"""
    log_print(f"    Treinando {classifier_name}...", log_file)

    # Treinar
    classifier.fit(X_train, y_train)

    # Predições
    y_pred = classifier.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results = {
        'classifier': classifier_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_test': y_test,
        'y_pred': y_pred
    }

    log_print(f"      Accuracy: {accuracy:.4f}", log_file)
    log_print(f"      Precision: {precision:.4f}", log_file)
    log_print(f"      Recall: {recall:.4f}", log_file)
    log_print(f"      F1-Score: {f1:.4f}", log_file)

    return results

def process_dataset(dataset_path, checkpoint_manager, log_file):
    """Processa um dataset completo"""
    dataset_name = os.path.basename(dataset_path).replace('.xlsx', '')

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSANDO DATASET: {dataset_name}", log_file)
    log_print(f"{'='*80}", log_file)

    # Carregar dataset
    log_print(f"\nCarregando {dataset_path}...", log_file)
    df = pd.read_excel(dataset_path)
    log_print(f"Dataset carregado: {len(df)} linhas", log_file)

    # Verificar coluna de classe
    class_column = 'Class' if 'Class' in df.columns else 'Classe'
    if class_column not in df.columns:
        log_print(f"ERRO: Coluna de classe não encontrada!", log_file)
        return

    all_results = []

    # Para cada combinação de campos
    for combo_name, fields in FIELD_COMBINATIONS.items():
        log_print(f"\n{'-'*80}", log_file)
        log_print(f"COMBINAÇÃO: {combo_name} - Campos: {fields}", log_file)
        log_print(f"{'-'*80}", log_file)

        checkpoint_key = f"{dataset_name}_{combo_name}"

        if checkpoint_manager.is_completed(checkpoint_key):
            log_print(f"  [CHECKPOINT] Combinação já processada, pulando...", log_file)
            continue

        # Combinar campos
        log_print(f"  Combinando campos...", log_file)
        texts = combine_fields(df, fields, use_processed=True)
        y = df[class_column].astype(str).to_numpy()

        # Criar embeddings TF-IDF
        log_print(f"  Gerando embeddings TF-IDF...", log_file)
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        log_print(f"  Dimensão dos embeddings: {X.shape}", log_file)

        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        log_print(f"  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}", log_file)

        # Treinar cada classificador
        for clf_name, clf in CLASSIFIERS.items():
            results = train_and_evaluate(
                X_train, X_test, y_train, y_test, 
                clf_name, clf, log_file
            )

            results['dataset'] = dataset_name
            results['combination'] = combo_name
            results['fields'] = fields
            results['embedding_dim'] = X.shape[1]
            all_results.append(results)

        # Salvar checkpoint
        checkpoint_manager.mark_completed(checkpoint_key)
        log_print(f"  [CHECKPOINT] Progresso salvo para {combo_name}", log_file)

    return all_results

def save_results_summary(all_results, log_file):
    """Salva resumo dos resultados"""
    log_print(f"\n\n{'='*80}", log_file)
    log_print(f"RESUMO GERAL DOS RESULTADOS", log_file)
    log_print(f"{'='*80}\n", log_file)

    # Criar DataFrame com resultados
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Dataset': result['dataset'],
            'Combination': result['combination'],
            'Classifier': result['classifier'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'Embedding_Dim': result['embedding_dim']
        })

    df_summary = pd.DataFrame(summary_data)

    # Salvar CSV para plotagem
    output_csv = os.path.join(RESULTS_DIR, 'resultados_tfidf.csv')
    df_summary.to_csv(output_csv, index=False)
    log_print(f"Resultados salvos em: {output_csv}\n", log_file)

    # Imprimir tabela formatada
    log_print(df_summary.to_string(index=False), log_file)

    # Melhores resultados por dataset e combinação
    log_print(f"\n\n{'='*80}", log_file)
    log_print(f"MELHORES RESULTADOS POR COMBINAÇÃO", log_file)
    log_print(f"{'='*80}\n", log_file)

    for dataset in df_summary['Dataset'].unique():
        log_print(f"\nDataset: {dataset}", log_file)
        log_print(f"{'-'*80}", log_file)

        dataset_results = df_summary[df_summary['Dataset'] == dataset]

        for combo in FIELD_COMBINATIONS.keys():
            combo_results = dataset_results[dataset_results['Combination'] == combo]
            if len(combo_results) > 0:
                best = combo_results.loc[combo_results['F1-Score'].idxmax()]
                log_print(f"  {combo:20s} | Melhor: {best['Classifier']:20s} | "
                         f"F1: {best['F1-Score']:.4f} | Acc: {best['Accuracy']:.4f}", log_file)

    return df_summary

def main():
    """Função principal"""
    # Arquivo de log
    log_file = os.path.join(RESULTS_DIR, f'log_tfidf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    log_print(f"{'='*80}", log_file)
    log_print(f"TREINAMENTO COM TF-IDF EMBEDDINGS", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

    # Checkpoint manager
    checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_tfidf.pkl')
    checkpoint_manager = CheckpointManager(checkpoint_file)

    all_results = []

    # Processar cada dataset
    for dataset in DATASETS:
        if os.path.exists(dataset):
            results = process_dataset(dataset, checkpoint_manager, log_file)
            if results:
                all_results.extend(results)
        else:
            log_print(f"\nAVISO: Dataset {dataset} não encontrado, pulando...", log_file)

    # Salvar resumo final
    if all_results:
        save_results_summary(all_results, log_file)

        # Salvar resultados completos em pickle
        results_pkl = os.path.join(RESULTS_DIR, 'resultados_completos_tfidf.pkl')
        with open(results_pkl, 'wb') as f:
            pickle.dump(all_results, f)
        log_print(f"\nResultados completos salvos em: {results_pkl}", log_file)

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSAMENTO CONCLUÍDO!", log_file)
    log_print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()
