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
import time
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

# Modelo GTE-ModernBERT-Base - ALIBABA NLP (MODERN ARCHITECTURE!)
GTE_MODEL = 'Alibaba-NLP/gte-modernbert-base'
BATCH_SIZE = 96  # 149M params, arquitetura moderna, batch grande
EMBEDDING_DIM = 768  # Dimensão padrão (suporta Matryoshka)
MAX_SEQ_LENGTH = 1024  # CONTEXTO LONGO! (suporta até 8192)

# Diretórios
CHECKPOINT_DIR = 'checkpoints_gte'
RESULTS_DIR = 'results_gte'
EMBEDDINGS_DIR = 'embeddings_gte'
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

def get_gte_embeddings(texts, model, batch_size, device, log_file=None):
    """
    Gera embeddings GTE-ModernBERT - ARQUITETURA MODERNIZADA
    """
    log_print(f"    Total de textos: {len(texts)}", log_file)
    log_print(f"    Batch size: {batch_size}", log_file)
    log_print(f"    Device: {device}", log_file)

    # GARANTIR que modelo está na GPU
    if device == 'cuda' and not next(model.parameters()).is_cuda:
        log_print(f"    ⚠️  Movendo modelo para GPU...", log_file)
        model = model.to(device)

    # Verificar VRAM
    if device == 'cuda':
        before_mem = torch.cuda.memory_allocated(0) / 1e9
        log_print(f"    VRAM antes: {before_mem:.2f} GB", log_file)

    # ModernBERT encoding (RoPE, local-global attention, flash attention)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )

    if device == 'cuda':
        after_mem = torch.cuda.memory_allocated(0) / 1e9
        log_print(f"    VRAM depois: {after_mem:.2f} GB", log_file)

    log_print(f"    ✓ Embeddings gerados: {embeddings.shape}", log_file)

    # Verificar NaN ou Inf
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        nan_count = np.isnan(embeddings).any(axis=1).sum()
        log_print(f"    ⚠️  Limpando {nan_count} embeddings com NaN/Inf", log_file)
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    return embeddings

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier_name, classifier, log_file):
    """Treina e avalia classificador"""
    log_print(f"    Treinando {classifier_name}...", log_file)

    if np.isnan(X_train).any() or np.isinf(X_train).any():
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(X_test).any() or np.isinf(X_test).any():
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

        # Verificar cache
        if os.path.exists(embeddings_file):
            log_print(f"  [CHECKPOINT] Carregando embeddings salvos...", log_file)
            X = np.load(embeddings_file)
            texts = combine_fields(df, fields, use_processed=True)
            y = df[class_column].values

            if np.isnan(X).any():
                log_print(f"  ⚠️  Cache corrompido, regenerando...", log_file)
                X = None
        else:
            X = None

        if X is None:
            if checkpoint_manager.is_completed(checkpoint_key):
                log_print(f"  [CHECKPOINT] Combinação já processada, pulando...", log_file)
                continue

            log_print(f"  Combinando campos...", log_file)
            texts = combine_fields(df, fields, use_processed=True)
            y = df[class_column].values

            log_print(f"  Gerando embeddings GTE-ModernBERT (USANDO GPU!)...", log_file)
            log_print(f"    Modelo: {GTE_MODEL}", log_file)

            X = get_gte_embeddings(texts, model, BATCH_SIZE, device, log_file=log_file)

            np.save(embeddings_file, X)
            log_print(f"  Embeddings salvos em: {embeddings_file}", log_file)

            if device == 'cuda':
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(0) / 1e9
                log_print(f"  ✓ Cache GPU limpo (usando {allocated:.2f} GB)", log_file)

        log_print(f"  Dimensão dos embeddings: {X.shape}", log_file)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        log_print(f"  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}", log_file)

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

    output_csv = os.path.join(RESULTS_DIR, 'resultados_gte.csv')
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
    log_file = os.path.join(RESULTS_DIR, f'log_gte_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    log_print(f"{'='*80}", log_file)
    log_print(f"TREINAMENTO COM GTE-MODERNBERT-BASE", log_file)
    log_print(f"Alibaba-NLP | 149M params | Dim: 768 | MODERN ARCHITECTURE", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

    if not torch.cuda.is_available():
        log_print(f"\n❌ ERRO: CUDA não disponível!", log_file)
        return

    device = 'cuda'
    log_print(f"\n✅ Dispositivo: {device}", log_file)
    log_print(f"✅ GPU: {torch.cuda.get_device_name(0)}", log_file)

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log_print(f"✅ Memória GPU: {total_mem:.2f} GB", log_file)

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    log_print(f"✅ Otimizações CUDA ativadas", log_file)

    log_print(f"\n📥 Carregando GTE-ModernBERT: {GTE_MODEL}", log_file)
    log_print(f"⚠️  ModernBERT = RoPE + Local-Global Attention + Flash Attention!", log_file)

    model = SentenceTransformer(GTE_MODEL, device=device, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH

    model_device = next(model.parameters()).device
    log_print(f"\n✓ Modelo carregado!", log_file)
    log_print(f"✓ Device: {model_device}", log_file)
    log_print(f"✓ Dimensão: {model.get_sentence_embedding_dimension()}", log_file)
    log_print(f"✓ Max seq length: {model.max_seq_length} (suporta 8192!)", log_file)
    log_print(f"✓ Batch size: {BATCH_SIZE}", log_file)
    log_print(f"✓ Arquitetura: 22 layers ModernBERT", log_file)
    log_print(f"✓ Features: RoPE, alternating attention, unpadding", log_file)

    if str(model_device) != 'cuda:0':
        model = model.to(device)
        log_print(f"✓ Modelo movido para GPU", log_file)

    allocated = torch.cuda.memory_allocated(0) / 1e9
    log_print(f"\n📊 VRAM alocada: {allocated:.2f} GB", log_file)

    # Teste de velocidade
    log_print(f"\n🧪 Teste de velocidade...", log_file)
    test_texts = ["modernbert arquitetura moderna"] * 200
    start = time.time()
    _ = model.encode(test_texts, batch_size=BATCH_SIZE, show_progress_bar=False, device=device)
    tempo = time.time() - start
    velocidade = 200 / tempo
    log_print(f"   200 textos em {tempo:.2f}s = {velocidade:.1f} txt/s", log_file)

    if velocidade < 25:
        log_print(f"   ⚠️  Velocidade abaixo do esperado", log_file)
    else:
        log_print(f"   ✅ Velocidade EXCELENTE (arquitetura moderna)!", log_file)
        tempo_estimado = 11902 / velocidade / 60
        log_print(f"   Tempo estimado: ~{tempo_estimado:.1f} minutos", log_file)

    checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_gte.pkl')
    checkpoint_manager = CheckpointManager(checkpoint_file)

    all_results = []

    for dataset in DATASETS:
        if os.path.exists(dataset):
            results = process_dataset(dataset, model, device, checkpoint_manager, log_file)
            if results:
                all_results.extend(results)
        else:
            log_print(f"\nAVISO: Dataset {dataset} não encontrado", log_file)

    if all_results:
        save_results_summary(all_results, log_file)

        results_pkl = os.path.join(RESULTS_DIR, 'resultados_completos_gte.pkl')
        with open(results_pkl, 'wb') as f:
            pickle.dump(all_results, f)
        log_print(f"\nResultados salvos em: {results_pkl}", log_file)

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSAMENTO CONCLUÍDO!", log_file)
    log_print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()
