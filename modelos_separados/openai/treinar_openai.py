import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from openai import OpenAI
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

# Parâmetros OpenAI
OPENAI_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50  # Reduzido para 50 para mais estabilidade
MAX_TOKENS_PER_TEXT = 8192

# Diretórios
CHECKPOINT_DIR = 'checkpoints_openai'
RESULTS_DIR = 'results_openai'
EMBEDDINGS_DIR = 'embeddings_openai'
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
                if text:  # Só adiciona se não estiver vazio
                    combined.append(text)

        # Se não houver texto, usar placeholder
        final_text = ' '.join(combined) if combined else 'sem texto'
        texts.append(final_text)

    return texts

def clean_text(text):
    """Limpa e valida texto para a API OpenAI"""
    if not text or pd.isna(text):
        return 'sem texto'

    text = str(text).strip()

    if not text:
        return 'sem texto'

    # Remover quebras de linha e caracteres especiais problemáticos
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Remove espaços múltiplos

    # Truncar se muito longo
    if len(text) > 8000:
        text = text[:8000]

    return text if text else 'sem texto'

def get_openai_embeddings_batch(texts, batch_size=50, log_file=None):
    """
    Gera embeddings em batches para economizar chamadas de API
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    log_print(f"    Total de textos: {len(texts)}", log_file)
    log_print(f"    Processando em {total_batches} batches de até {batch_size} textos", log_file)

    for i in tqdm(range(0, len(texts), batch_size), desc="  Gerando embeddings OpenAI"):
        batch_texts = texts[i:i+batch_size]

        # Limpar e validar todos os textos
        batch_texts = [clean_text(text) for text in batch_texts]

        # Verificar se há textos válidos
        if not any(batch_texts):
            log_print(f"    Batch {i//batch_size + 1}: Todos os textos vazios, usando vetores zero", log_file)
            all_embeddings.extend([[0.0] * EMBEDDING_DIMENSION] * len(batch_texts))
            continue

        retries = 3
        success = False

        for attempt in range(retries):
            try:
                # Fazer request em batch
                response = client.embeddings.create(
                    input=batch_texts,
                    model=OPENAI_MODEL,
                    dimensions=EMBEDDING_DIMENSION
                )

                # Extrair embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Log de uso de tokens
                if hasattr(response, 'usage'):
                    log_print(f"      Batch {i//batch_size + 1}/{total_batches}: "
                             f"{response.usage.total_tokens} tokens usados", log_file)

                success = True
                break

            except Exception as e:
                error_msg = str(e)
                log_print(f"    Tentativa {attempt + 1}/{retries} falhou no batch {i//batch_size + 1}: {error_msg}", log_file)

                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Última tentativa: processar individualmente
                    log_print(f"    Processando textos individuais para batch {i//batch_size + 1}...", log_file)

                    for idx, text in enumerate(batch_texts):
                        try:
                            response = client.embeddings.create(
                                input=[text],
                                model=OPENAI_MODEL,
                                dimensions=EMBEDDING_DIMENSION
                            )
                            all_embeddings.append(response.data[0].embedding)
                        except Exception as e2:
                            log_print(f"      Erro no texto {idx}: {str(e2)[:100]}, usando vetor zero", log_file)
                            all_embeddings.append([0.0] * EMBEDDING_DIMENSION)

                    success = True

        if success:
            # Rate limiting gentil
            time.sleep(0.2)

    return np.array(all_embeddings)

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier_name, classifier, log_file):
    """Treina e avalia classificador"""
    log_print(f"    Treinando {classifier_name}...", log_file)

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
        return

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
                log_print(f"  Aviso: {empty_count} textos vazios encontrados (serão tratados)", log_file)

            # Gerar embeddings OpenAI em batches
            log_print(f"  Gerando embeddings OpenAI ({OPENAI_MODEL})...", log_file)
            log_print(f"    Dimensão: {EMBEDDING_DIMENSION}", log_file)
            log_print(f"    Batch size: {BATCH_SIZE}", log_file)

            X = get_openai_embeddings_batch(texts, batch_size=BATCH_SIZE, log_file=log_file)

            # Verificar se há vetores zero demais
            zero_vectors = np.sum(np.all(X == 0, axis=1))
            if zero_vectors > 0:
                log_print(f"  Aviso: {zero_vectors} embeddings são vetores zero", log_file)

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

    output_csv = os.path.join(RESULTS_DIR, 'resultados_openai.csv')
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
    log_file = os.path.join(RESULTS_DIR, f'log_openai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    log_print(f"{'='*80}", log_file)
    log_print(f"TREINAMENTO COM OPENAI EMBEDDINGS (text-embedding-3-small)", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

    # Verificar API key
    try:
        test_response = client.embeddings.create(
            input=["teste de conexão"],
            model=OPENAI_MODEL,
            dimensions=EMBEDDING_DIMENSION
        )
        log_print(f"\n✓ API Key OpenAI configurada corretamente", log_file)
        log_print(f"✓ Modelo: {OPENAI_MODEL}", log_file)
        log_print(f"✓ Dimensão dos embeddings: {EMBEDDING_DIMENSION}", log_file)
    except Exception as e:
        log_print(f"\nERRO: Não foi possível conectar à API OpenAI", log_file)
        log_print(f"Mensagem: {str(e)}", log_file)
        log_print(f"\nConfigure a variável de ambiente OPENAI_API_KEY:", log_file)
        log_print(f"  export OPENAI_API_KEY='sua-chave-aqui'", log_file)
        return

    checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_openai.pkl')
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

        results_pkl = os.path.join(RESULTS_DIR, 'resultados_completos_openai.pkl')
        with open(results_pkl, 'wb') as f:
            pickle.dump(all_results, f)
        log_print(f"\nResultados completos salvos em: {results_pkl}", log_file)

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSAMENTO CONCLUÍDO!", log_file)
    log_print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()
