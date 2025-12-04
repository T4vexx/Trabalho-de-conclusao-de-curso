import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Tentar importar cuML (GPU)
try:
    import cuml
    from cuml.svm import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ RAPIDS cuML detectado - Classificadores rodarão na GPU!")
except ImportError:
    from sklearn.svm import SVC as cuSVC
    from sklearn.ensemble import RandomForestClassifier as cuRF
    from sklearn.linear_model import LogisticRegression as cuLR
    GPU_AVAILABLE = False
    print("⚠️  RAPIDS cuML não instalado - Classificadores rodarão na CPU")

# --- CONFIGURAÇÕES ---
DATASETS = [
    '../../../bases_preprocessados/dataset_sem_stopwords_lemmatizado.xlsx',
    '../../../bases_preprocessados/dataset_sem_caracteres_especiais_lemmatizado.xlsx'
]
BEST_COMBINATION = {
    'completo': ['Titulo', 'Subtitulo', 'Noticia']
}
# OpenAI API
OPENAI_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50
# ATENÇÃO: É uma má prática colocar a chave de API diretamente no código.
# Use variáveis de ambiente para segurança.


# --- HIPERPARÂMETROS ---
PARAM_GRIDS_FAST = {
    'SVM': {
        'C': [0.1, 1.0, 10.0, 100.0], 'kernel': ['linear'], 'class_weight': [None, 'balanced']
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300], 'max_depth': [20, 30, None],
        'min_samples_split': [2, 5], 'max_features': ['sqrt', 'log2']
    },
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0, 100.0], 'penalty': ['l2'],
        # CORREÇÃO: Remover parâmetros incompatíveis com cuML
        **({'solver': ['lbfgs'], 'class_weight': [None, 'balanced']} if not GPU_AVAILABLE else {}),
        'max_iter': [2000]
    }
}

PARAM_DISTRIBUTIONS_FAST = {
    'SVM': {
        'C': uniform(0.1, 100), 'kernel': ['linear'], 'class_weight': [None, 'balanced']
    },
    'RandomForest': {
        'n_estimators': randint(100, 400),
        # CORREÇÃO: cuML RandomForest não aceita 'None' para max_depth.
        'max_depth': [20, 30, 50],
        'min_samples_split': randint(2, 11),
        'max_features': ['sqrt', 'log2']
    },
    'LogisticRegression': {
        'C': uniform(0.1, 100), 'penalty': ['l2'],
        # CORREÇÃO: Remover parâmetros incompatíveis com cuML
        **({'solver': ['lbfgs'], 'class_weight': [None, 'balanced']} if not GPU_AVAILABLE else {}),
        'max_iter': [2000]
    }
}

# --- DIRETÓRIOS ---
EMBEDDINGS_DIR = 'embeddings_openai_hypertuning'
RESULTS_DIR = 'results_openai_hypertuning_gpu'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- FUNÇÕES AUXILIARES (sem alterações) ---
def log_print(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f: f.write(message + '\n')
def combine_fields(df, fields):
    texts = []
    for _, row in df.iterrows():
        combined = [str(row[field]).strip() for field in fields if field in df.columns and pd.notna(row[field])]
        texts.append(' '.join(combined) if combined else 'sem texto')
    return texts
def clean_text(text):
    text = str(text).strip().replace('\n', ' ').replace('\r', ' ')
    return ' '.join(text.split())[:8000] or 'sem texto'
def get_openai_embeddings_batch(texts, log_file):
    all_embeddings, total_batches = [], (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    log_print(f"    Processando {len(texts)} textos em {total_batches} batches...", log_file)
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="    OpenAI API"):
        batch_texts = [clean_text(t) for t in texts[i:i+BATCH_SIZE]]
        try:
            response = client.embeddings.create(input=batch_texts, model=OPENAI_MODEL, dimensions=EMBEDDING_DIMENSION)
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            log_print(f"    ❌ Erro no batch {i//BATCH_SIZE + 1}: {e}", log_file)
            all_embeddings.extend([[0.0] * EMBEDDING_DIMENSION] * len(batch_texts))
        time.sleep(0.1)
    return np.array(all_embeddings)
def convert_to_gpu(X_train, X_test, y_train, y_test, log_file):
    if GPU_AVAILABLE:
        log_print("    Transferindo dados para GPU...", log_file)
        return cp.asarray(X_train), cp.asarray(X_test), cp.asarray(y_train), cp.asarray(y_test)
    return X_train, X_test, y_train, y_test

# --- FUNÇÕES DE TUNING (COM CORREÇÕES) ---
def hyperparameter_tuning_manual(X_train, X_test, y_train, y_test, classifier_name, param_grid, log_file, use_gpu=True):
    log_print(f"\n{'='*80}\nGRID SEARCH: {classifier_name}", log_file)
    log_print(f"🚀 GPU (cuML)" if use_gpu and GPU_AVAILABLE else "💻 CPU (sklearn)", log_file)
    log_print(f"{'='*80}", log_file)
    X_train_proc, X_test_proc, y_train_proc, y_test_proc = convert_to_gpu(X_train, X_test, y_train, y_test, log_file) if use_gpu else (X_train, X_test, y_train, y_test)
    from itertools import product
    param_keys, param_values = list(param_grid.keys()), list(param_grid.values())
    all_combinations = list(product(*param_values))
    log_print(f"Testando {len(all_combinations)} combinações...", log_file)
    best_f1, best_params, best_model = -1, None, None
    for i, param_combo in enumerate(all_combinations):
        params = dict(zip(param_keys, param_combo))
        try:
            # CORREÇÃO: Não passar random_state para LogisticRegression do cuML
            if classifier_name in ['SVM', 'RandomForest']:
                model = (cuSVC if classifier_name == 'SVM' else cuRF)(random_state=42, **params)
            else:
                model = cuLR(**params)
            model.fit(X_train_proc, y_train_proc)
            y_pred = model.predict(X_test_proc)
            y_pred_cpu = cp.asnumpy(y_pred) if GPU_AVAILABLE and use_gpu else y_pred
            y_test_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc
            f1 = f1_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0)
            if f1 > best_f1:
                best_f1, best_params, best_model = f1, params, model
                log_print(f"  [{i+1}/{len(all_combinations)}] ✓ Melhor! F1={f1:.4f} | {params}", log_file)
        except Exception as e:
            log_print(f"  [{i+1}/{len(all_combinations)}] ❌ Erro em {params}: {str(e)[:100]}", log_file)
    log_print(f"\n✅ Concluído!", log_file)
    # CORREÇÃO: Adicionar verificação para evitar crash
    if best_params is None:
        log_print(f"\n❌ Nenhuma combinação funcionou para {classifier_name}.", log_file)
        return {'method': 'GridSearch', 'classifier': classifier_name, 'test_f1': 0, 'error': 'All combinations failed'}
    log_print(f"\n🏆 MELHORES:\n" + "\n".join([f"  • {p}: {v}" for p, v in best_params.items()]), log_file)
    y_pred_final = best_model.predict(X_test_proc)
    y_pred_final_cpu = cp.asnumpy(y_pred_final) if GPU_AVAILABLE and use_gpu else y_pred_final
    y_test_final_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc
    accuracy = accuracy_score(y_test_final_cpu, y_pred_final_cpu)
    f1 = f1_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)
    log_print(f"\n📈 TESTE:\n  • Accuracy:  {accuracy:.4f}\n  • F1-Score:  {f1:.4f}", log_file)
    return {'method': 'GridSearch', 'classifier': classifier_name, 'best_params': best_params, 'test_accuracy': accuracy, 'test_f1': f1, 'best_model': best_model, 'used_gpu': GPU_AVAILABLE and use_gpu}

def hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, classifier_name, param_distributions, n_iter, log_file, use_gpu=True):
    log_print(f"\n{'='*80}\nRANDOM SEARCH: {classifier_name}", log_file)
    log_print(f"🚀 GPU (cuML)" if use_gpu and GPU_AVAILABLE else "💻 CPU (sklearn)", log_file)
    log_print(f"{'='*80}", log_file)
    X_train_proc, X_test_proc, y_train_proc, y_test_proc = convert_to_gpu(X_train, X_test, y_train, y_test, log_file) if use_gpu else (X_train, X_test, y_train, y_test)
    log_print(f"Testando {n_iter} combinações...", log_file)
    best_f1, best_params, best_model = -1, None, None
    np.random.seed(42)
    for i in range(n_iter):
        params = {k: v.rvs() if hasattr(v, 'rvs') else np.random.choice(v) for k, v in param_distributions.items()}
        try:
            # CORREÇÃO: Não passar random_state para LogisticRegression do cuML
            if classifier_name in ['SVM', 'RandomForest']:
                model = (cuSVC if classifier_name == 'SVM' else cuRF)(random_state=42, **params)
            else:
                model = cuLR(**params)
            model.fit(X_train_proc, y_train_proc)
            y_pred = model.predict(X_test_proc)
            y_pred_cpu = cp.asnumpy(y_pred) if GPU_AVAILABLE and use_gpu else y_pred
            y_test_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc
            f1 = f1_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0)
            if f1 > best_f1:
                best_f1, best_params, best_model = f1, params, model
                log_print(f"  [{i+1}/{n_iter}] ✓ Melhor! F1={f1:.4f}", log_file)
        except Exception as e:
            log_print(f"  [{i+1}/{n_iter}] ❌ Erro em {params}: {str(e)[:100]}", log_file)
    log_print(f"\n✅ Concluído!", log_file)
    # CORREÇÃO: Adicionar verificação para evitar crash
    if best_params is None:
        log_print(f"\n❌ Nenhuma combinação funcionou para {classifier_name}.", log_file)
        return {'method': 'RandomSearch', 'classifier': classifier_name, 'test_f1': 0, 'error': 'All combinations failed'}
    y_pred_final = best_model.predict(X_test_proc)
    y_pred_final_cpu = cp.asnumpy(y_pred_final) if GPU_AVAILABLE and use_gpu else y_pred_final
    y_test_final_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc
    accuracy = accuracy_score(y_test_final_cpu, y_pred_final_cpu)
    f1 = f1_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)
    log_print(f"\n📈 TESTE: Acc={accuracy:.4f} | F1={f1:.4f}", log_file)
    return {'method': 'RandomSearch', 'classifier': classifier_name, 'best_params': best_params, 'test_accuracy': accuracy, 'test_f1': f1, 'best_model': best_model, 'used_gpu': GPU_AVAILABLE and use_gpu}

# --- FUNÇÃO PRINCIPAL (COM PEQUENAS MELHORIAS) ---
def main():
    log_file = os.path.join(RESULTS_DIR, f'log_openai_hypertuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    log_print(f"{'='*80}\nOPENAI: HYPERPARAMETER TUNING COM GPU\nModelo: {OPENAI_MODEL}\nGPU Status: cuML {'✅' if GPU_AVAILABLE else '❌'}\n{'='*80}", log_file)
    try:
        client.embeddings.create(input=["teste"], model=OPENAI_MODEL, dimensions=EMBEDDING_DIMENSION)
        log_print("\n✅ API OpenAI OK", log_file)
    except Exception as e:
        log_print(f"\n❌ Erro API: {e}", log_file)
        return
    all_results = []
    for dataset_path in DATASETS:
        if not os.path.exists(dataset_path): continue
        dataset_name = os.path.basename(dataset_path).replace('.xlsx', '')
        log_print(f"\n\n{'#'*80}\nDATASET: {dataset_name}\n{'#'*80}", log_file)
        try:
            df = pd.read_excel(dataset_path, engine='openpyxl')
        except ImportError:
            log_print("A biblioteca 'openpyxl' é necessária. Instale com: conda install openpyxl", log_file)
            return
        class_column = 'Class' if 'Class' in df.columns else 'Classe'
        y = df[class_column].values
        for combo_name, fields in BEST_COMBINATION.items():
            cache_file = os.path.join(EMBEDDINGS_DIR, f'embeddings_{dataset_name}_{combo_name}.npy')
            if os.path.exists(cache_file):
                log_print(f"\n✓ Usando embeddings do cache para: {combo_name}", log_file)
                X = np.load(cache_file)
            else:
                log_print(f"\nGerando embeddings para: {combo_name}", log_file)
                texts = combine_fields(df, fields)
                X = get_openai_embeddings_batch(texts, log_file)
                np.save(cache_file, X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            # Executar todos os tunings
            for res in [
                hyperparameter_tuning_manual(X_train, X_test, y_train, y_test, 'SVM', PARAM_GRIDS_FAST['SVM'], log_file),
                hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, 'SVM', PARAM_DISTRIBUTIONS_FAST['SVM'], 15, log_file),
                hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, 'RandomForest', PARAM_DISTRIBUTIONS_FAST['RandomForest'], 20, log_file),
                hyperparameter_tuning_manual(X_train, X_test, y_train, y_test, 'LogisticRegression', PARAM_GRIDS_FAST['LogisticRegression'], log_file),
                hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, 'LogisticRegression', PARAM_DISTRIBUTIONS_FAST['LogisticRegression'], 15, log_file)
            ]:
                res.update({'dataset': dataset_name, 'combination': combo_name})
                all_results.append(res)
    # Resumo
    all_results = [res for res in all_results if 'error' not in res]
    if not all_results:
        log_print("\nNenhum resultado gerado com sucesso.", log_file)
        return
    summary_data = [{'Dataset': r.get('dataset'), 'Method': r.get('method'), 'Classifier': r.get('classifier'), 'Test_F1': r.get('test_f1'), 'Test_Accuracy': r.get('test_accuracy'), 'Used_GPU': r.get('used_gpu'), 'Best_Params': str(r.get('best_params'))} for r in all_results]
    df_summary = pd.DataFrame(summary_data).sort_values(by='Test_F1', ascending=False)
    output_csv = os.path.join(RESULTS_DIR, 'melhores_hiperparametros_openai_gpu.csv')
    df_summary.to_csv(output_csv, index=False)
    log_print(f"\n{df_summary.to_string(index=False)}", log_file)
    log_print(f"\n✅ Resultados salvos em: {output_csv}", log_file)
    # Salvar modelos
    models_file = os.path.join(RESULTS_DIR, 'best_models_openai_gpu.pkl')
    with open(models_file, 'wb') as f: pickle.dump(all_results, f)
    log_print(f"\n{'='*80}\nCONCLUÍDO!\n{'='*80}", log_file)

if __name__ == "__main__":
    main()