import pandas as pd
import numpy as np
import pickle
import os
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
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

# Modelo SERAFIM para português
SERAFIM_MODEL = 'PORTULAN/serafim-900m-portuguese-pt-sentence-encoder'
BATCH_SIZE = 32  # Sentence-transformers é mais eficiente, pode usar batch maior
EMBEDDING_DIM = 1536  # Dimensão do SERAFIM

# Diretórios
CHECKPOINT_DIR = 'checkpoints_serafim'
RESULTS_DIR = 'results_serafim'
EMBEDDINGS_DIR = 'embeddings_serafim'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

class CheckpointManager:
    """Gerenciador de checkpoints"""

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
    """Imprime na tela e salva no log"""
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
                text = str(row[field_name]).strip()
                if text:
                    combined.append(text)

        final_text = ' '.join(combined) if combined else 'sem texto'
        texts.append(final_text)

    return texts

def get_serafim_embeddings(texts, model, batch_size=32, device=None, log_file=None):
    """
    Gera embeddings SERAFIM em batches usando GPU se disponível
    Sentence-Transformers já tem otimizações internas para batch processing
    """
    log_print(f"    Total de textos: {len(texts)}", log_file)
    log_print(f"    Processando em batches de {batch_size}...", log_file)

    # Sentence-Transformers já faz batch processing otimizado
    # show_progress_bar=True para ver progresso
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalização L2 automática
    )

    log_print(f"    ✓ Embeddings gerados: {embeddings.shape}", log_file)

    # Verificar NaN ou Inf
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        nan_count = np.isnan(embeddings).any(axis=1).sum()
        inf_count = np.isinf(embeddings).any(axis=1).sum()
        log_print(f"    ⚠️  Limpando {nan_count} NaN e {inf_count} Inf", log_file)
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    return embeddings

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier_name, classifier, log_file):
    """Treina e avalia classificador"""
    log_print(f"    Treinando {classifier_name}...", log_file)

    # Verificação extra de segurança
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        log_print(f"      ⚠️  Limpando dados de treino...", log_file)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(X_test).any() or np.isinf(X_test).any():
        log_print(f"      ⚠️  Limpando dados de teste...", log_file)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

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

def process_dataset(dataset_path, model, device, checkpoint_manager, log_file):
    """Processa um dataset"""
    dataset_name = os.path.basename(dataset_path).replace('.xlsx', '')

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSANDO DATASET: {dataset_name}", log_file)
    log_print(f"{'='*80}", log_file)

    log_print(f"\nCarregando {dataset_path}...", log_file)
    df = pd.read_excel(dataset_path)
    log_print(f"Dataset carregado: {len(df)} linhas", log_file)

    class_column = 'Class' if 'Class' in df.columns else 'Classe'
    if class_column not in df.columns:
        log_print(f"ERRO: Coluna de classe não encontrada!", log_file)
        return []

    all_results = []

    for combo_name, fields in FIELD_COMBINATIONS.items():
        log_print(f"\n{'-'*80}", log_file)
        log_print(f"COMBINAÇÃO: {combo_name} - Campos: {fields}", log_file)
        log_print(f"{'-'*80}", log_file)

        checkpoint_key = f"{dataset_name}_{combo_name}"
        embeddings_file = os.path.join(EMBEDDINGS_DIR, f'embeddings_{checkpoint_key}.npy')

        # Verificar se embeddings já foram gerados (cache)
        if os.path.exists(embeddings_file):
            log_print(f"  [CHECKPOINT] Carregando embeddings salvos...", log_file)
            X = np.load(embeddings_file)
            texts = combine_fields(df, fields, use_processed=True)
            y = df[class_column].values

            # Verificar integridade do cache
            if np.isnan(X).any() or np.isinf(X).any():
                log_print(f"  ⚠️  Cache corrompido, regenerando...", log_file)
                X = None

        else:
            X = None

        if X is None:
            if checkpoint_manager.is_completed(checkpoint_key):
                log_print(f"  [CHECKPOINT] Combinação já processada, pulando...", log_file)
                continue

            # Combinar campos
            log_print(f"  Combinando campos...", log_file)
            texts = combine_fields(df, fields, use_processed=True)
            y = df[class_column].values

            # Log de diagnóstico
            empty_count = sum(1 for t in texts if t == 'sem texto')
            if empty_count > 0:
                log_print(f"  Aviso: {empty_count} textos vazios encontrados", log_file)

            # Gerar embeddings SERAFIM
            log_print(f"  Gerando embeddings SERAFIM (usando {device})...", log_file)
            log_print(f"    Modelo: {SERAFIM_MODEL}", log_file)
            log_print(f"    Batch size: {BATCH_SIZE}", log_file)
            log_print(f"    Dimensão: {EMBEDDING_DIM}", log_file)

            X = get_serafim_embeddings(texts, model, batch_size=BATCH_SIZE, device=device, log_file=log_file)

            # Salvar embeddings em cache
            np.save(embeddings_file, X)
            log_print(f"  Embeddings salvos em: {embeddings_file}", log_file)

        log_print(f"  Dimensão dos embeddings: {X.shape}", log_file)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        log_print(f"  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}", log_file)

        # Treinar classificadores
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

    output_csv = os.path.join(RESULTS_DIR, 'resultados_serafim.csv')
    df_summary.to_csv(output_csv, index=False)
    log_print(f"Resultados salvos em: {output_csv}\n", log_file)

    log_print(df_summary.to_string(index=False), log_file)

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
    log_file = os.path.join(RESULTS_DIR, f'log_serafim_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    log_print(f"{'='*80}", log_file)
    log_print(f"TREINAMENTO COM SERAFIM EMBEDDINGS", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

    # Verificar GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_print(f"\nDispositivo: {device}", log_file)
    if torch.cuda.is_available():
        log_print(f"GPU: {torch.cuda.get_device_name(0)}", log_file)
        log_print(f"Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", log_file)

    # Carregar modelo SERAFIM
    log_print(f"\nCarregando modelo SERAFIM: {SERAFIM_MODEL}", log_file)
    log_print(f"Aguarde... (primeira vez pode demorar para baixar o modelo)", log_file)

    model = SentenceTransformer(SERAFIM_MODEL, device=device)

    log_print(f"✓ Modelo carregado com sucesso!", log_file)
    log_print(f"✓ Dimensão dos embeddings: {model.get_sentence_embedding_dimension()}", log_file)
    log_print(f"✓ Max sequence length: {model.max_seq_length}", log_file)

    checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_serafim.pkl')
    checkpoint_manager = CheckpointManager(checkpoint_file)

    all_results = []

    for dataset in DATASETS:
        if os.path.exists(dataset):
            results = process_dataset(dataset, model, device, checkpoint_manager, log_file)
            if results:
                all_results.extend(results)
        else:
            log_print(f"\nAVISO: Dataset {dataset} não encontrado, pulando...", log_file)

    if all_results:
        save_results_summary(all_results, log_file)

        results_pkl = os.path.join(RESULTS_DIR, 'resultados_completos_serafim.pkl')
        with open(results_pkl, 'wb') as f:
            pickle.dump(all_results, f)
        log_print(f"\nResultados completos salvos em: {results_pkl}", log_file)

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSAMENTO CONCLUÍDO!", log_file)
    log_print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()
