import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
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

# Parâmetros Gemini OTIMIZADOS PARA BATCH
GEMINI_MODEL = 'models/embedding-001'
MAX_TEXT_LENGTH = 10000  
BATCH_SIZE = 100  # Processa 100 textos por request
RATE_LIMIT_DELAY = 0.2  # Delay entre batches

# Diretórios
CHECKPOINT_DIR = 'checkpoints_gemini'
RESULTS_DIR = 'results_gemini'
EMBEDDINGS_DIR = 'embeddings_gemini'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def initialize_gemini_client():
    """Inicializa cliente Gemini com API key do ambiente"""
    try:
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("API key não encontrada")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"ERRO ao inicializar cliente Gemini: {str(e)}")
        print("Configure a variável de ambiente GEMINI_API_KEY ou GOOGLE_API_KEY")
        return False

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

def clean_text(text):
    """Limpa e valida texto para a API Gemini"""
    if not text or pd.isna(text):
        return 'sem texto'

    text = str(text).strip()

    if not text:
        return 'sem texto'

    # Remove espaços múltiplos
    text = ' '.join(text.split())

    # Truncar se muito longo
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    return text if text else 'sem texto'

def safe_normalize_l2(X):
    """
    Normalização L2 segura que trata vetores zero
    Vetores zero continuam zero (não viram NaN)
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # Evitar divisão por zero: onde norm == 0, manter vetor original (zero)
    norms = np.where(norms == 0, 1, norms)

    X_normalized = X / norms

    # Verificar se há NaN após normalização
    nan_count = np.isnan(X_normalized).any(axis=1).sum()
    if nan_count > 0:
        # Substituir linhas com NaN por vetores zero
        nan_mask = np.isnan(X_normalized).any(axis=1)
        X_normalized[nan_mask] = 0
        print(f"    ⚠️  {nan_count} vetores com NaN foram substituídos por zeros")

    return X_normalized

def get_gemini_embeddings_batch(texts, batch_size=100, log_file=None):
    """
    Gera embeddings EM BATCH - OTIMIZADO
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    log_print(f"    Total de textos: {len(texts)}", log_file)
    log_print(f"    Processando em {total_batches} batches de até {batch_size} textos", log_file)

    # Limpar todos os textos antes
    texts_cleaned = [clean_text(text) for text in texts]

    for i in tqdm(range(0, len(texts_cleaned), batch_size), desc="  Gerando embeddings Gemini"):
        batch_texts = texts_cleaned[i:i+batch_size]

        retries = 3
        success = False

        for attempt in range(retries):
            try:
                # USAR BATCH PROCESSING
                result = genai.embed_content(
                    model=GEMINI_MODEL,
                    content=batch_texts,
                    task_type="classification"
                )

                # Extrair embeddings do batch
                if isinstance(result['embedding'][0], list):
                    # Lista de listas
                    batch_embeddings = [np.array(emb) for emb in result['embedding']]
                else:
                    # Array único
                    batch_embeddings = [np.array(result['embedding'])]

                all_embeddings.extend(batch_embeddings)

                log_print(f"      Batch {i//batch_size + 1}/{total_batches}: "
                         f"{len(batch_texts)} textos processados", log_file)

                success = True
                break

            except Exception as e:
                error_msg = str(e)
                log_print(f"    Tentativa {attempt + 1}/{retries} falhou no batch {i//batch_size + 1}: {error_msg[:100]}", log_file)

                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    # Processar individualmente
                    log_print(f"    Processando textos individuais para batch {i//batch_size + 1}...", log_file)

                    for idx, text in enumerate(batch_texts):
                        try:
                            result = genai.embed_content(
                                model=GEMINI_MODEL,
                                content=text,
                                task_type="classification"
                            )
                            all_embeddings.append(np.array(result['embedding']))
                            time.sleep(0.05)
                        except Exception as e2:
                            log_print(f"      Erro no texto {idx}: {str(e2)[:100]}, usando vetor zero", log_file)
                            all_embeddings.append(np.zeros(768))

                    success = True

        if success:
            time.sleep(RATE_LIMIT_DELAY)

    return np.array(all_embeddings)

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier_name, classifier, log_file):
    """Treina e avalia classificador"""
    log_print(f"    Treinando {classifier_name}...", log_file)

    # Verificar se há NaN ou Inf
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        log_print(f"      ⚠️  AVISO: Dados de treino contêm NaN ou Inf, limpando...", log_file)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(X_test).any() or np.isinf(X_test).any():
        log_print(f"      ⚠️  AVISO: Dados de teste contêm NaN ou Inf, limpando...", log_file)
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

def process_dataset(dataset_path, checkpoint_manager, log_file):
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

            # Verificar se há NaN no cache
            if np.isnan(X).any():
                log_print(f"  ⚠️  Cache contém NaN, limpando...", log_file)
                X = np.nan_to_num(X, nan=0.0)
                # Re-salvar cache limpo
                np.save(embeddings_file, X)

        elif checkpoint_manager.is_completed(checkpoint_key):
            log_print(f"  [CHECKPOINT] Combinação já processada, pulando...", log_file)
            continue
        else:
            # Combinar campos
            log_print(f"  Combinando campos...", log_file)
            texts = combine_fields(df, fields, use_processed=True)
            y = df[class_column].values

            # Log de diagnóstico
            empty_count = sum(1 for t in texts if t == 'sem texto')
            if empty_count > 0:
                log_print(f"  Aviso: {empty_count} textos vazios encontrados", log_file)

            # Gerar embeddings Gemini EM BATCH
            log_print(f"  Gerando embeddings Gemini ({GEMINI_MODEL})...", log_file)
            log_print(f"    Batch size: {BATCH_SIZE} (OTIMIZADO)", log_file)

            X = get_gemini_embeddings_batch(texts, batch_size=BATCH_SIZE, log_file=log_file)

            # Normalizar embeddings com função segura
            log_print(f"    Normalizando embeddings (L2 norm)...", log_file)
            X = safe_normalize_l2(X)
            log_print(f"    ✓ Embeddings normalizados", log_file)

            # Verificar vetores zero APÓS normalização
            zero_vectors = np.sum(np.all(X == 0, axis=1))
            if zero_vectors > 0:
                log_print(f"  Aviso: {zero_vectors} embeddings são vetores zero após normalização", log_file)

            # Verificação final de NaN
            if np.isnan(X).any():
                nan_count = np.isnan(X).any(axis=1).sum()
                log_print(f"  ⚠️  ERRO: {nan_count} embeddings contêm NaN, limpando...", log_file)
                X = np.nan_to_num(X, nan=0.0)

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

    output_csv = os.path.join(RESULTS_DIR, 'resultados_gemini.csv')
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
    log_file = os.path.join(RESULTS_DIR, f'log_gemini_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    log_print(f"{'='*80}", log_file)
    log_print(f"TREINAMENTO COM GOOGLE GEMINI EMBEDDINGS (BATCH + NaN FIX)", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

    # Inicializar cliente Gemini
    log_print(f"\nInicializando cliente Gemini...", log_file)
    if not initialize_gemini_client():
        log_print(f"\nERRO: Não foi possível inicializar o cliente Gemini", log_file)
        return

    # Teste de conexão
    try:
        test_result = genai.embed_content(
            model=GEMINI_MODEL,
            content="teste de conexão",
            task_type="classification"
        )
        log_print(f"✓ API Gemini configurada corretamente", log_file)
        log_print(f"✓ Modelo: {GEMINI_MODEL}", log_file)
        log_print(f"✓ Dimensão dos embeddings: {len(test_result['embedding'])}", log_file)
        log_print(f"✓ Batch size: {BATCH_SIZE} textos por request", log_file)
        log_print(f"✓ Modo: BATCH + NaN SAFE", log_file)
    except Exception as e:
        log_print(f"\nERRO ao testar API: {str(e)}", log_file)
        return

    checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_gemini.pkl')
    checkpoint_manager = CheckpointManager(checkpoint_file)

    all_results = []

    for dataset in DATASETS:
        if os.path.exists(dataset):
            results = process_dataset(dataset, checkpoint_manager, log_file)
            if results:
                all_results.extend(results)
        else:
            log_print(f"\nAVISO: Dataset {dataset} não encontrado, pulando...", log_file)

    if all_results:
        save_results_summary(all_results, log_file)

        results_pkl = os.path.join(RESULTS_DIR, 'resultados_completos_gemini.pkl')
        with open(results_pkl, 'wb') as f:
            pickle.dump(all_results, f)
        log_print(f"\nResultados completos salvos em: {results_pkl}", log_file)

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSAMENTO CONCLUÍDO!", log_file)
    log_print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()
