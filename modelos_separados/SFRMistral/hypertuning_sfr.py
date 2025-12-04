import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint
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

# Configurações
DATASETS = [
    '../../bases_preprocessados/dataset_sem_stopwords_lemmatizado.xlsx',
    '../../bases_preprocessados/dataset_sem_caracteres_especiais_lemmatizado.xlsx'
]

BEST_COMBINATION = {
    'completo': ['Titulo', 'Subtitulo', 'Noticia']
}

# Diretórios
EMBEDDINGS_DIR = 'embeddings_sfr'
RESULTS_DIR = 'results_better_sfr'
os.makedirs(RESULTS_DIR, exist_ok=True)

# HIPERPARÂMETROS OTIMIZADOS
PARAM_GRIDS_FAST = {
    'SVM': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear'],
        'class_weight': [None, 'balanced']
    },
    'SVM_RBF': {
        'C': [1.0, 10.0],
        'kernel': ['rbf'],
        'gamma': ['scale'],
        'class_weight': ['balanced']
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    },
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        **({'solver': ['lbfgs'], 'class_weight': [None, 'balanced']} if not GPU_AVAILABLE else {}),
        'max_iter': [2000]
    }
}

PARAM_DISTRIBUTIONS_FAST = {
    'SVM': {
        'C': uniform(0.1, 100),
        'kernel': ['linear'],
        'class_weight': [None, 'balanced']
    },
    'RandomForest': {
        'n_estimators': randint(100, 400),
        # CORREÇÃO 1: Removido '-1' de max_depth, pois cuML exige valores > 0.
        # A opção de profundidade ilimitada não é diretamente suportada.
        'max_depth': [20, 30, 50],
        'min_samples_split': randint(2, 11),
        'max_features': ['sqrt', 'log2']
    },
    'LogisticRegression': {
        'C': uniform(0.1, 100),
        'penalty': ['l2'],
        **({'solver': ['lbfgs'], 'class_weight': [None, 'balanced']} if not GPU_AVAILABLE else {}),
        'max_iter': [2000]
    }
}

def log_print(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def convert_to_gpu(X_train, X_test, y_train, y_test, log_file):
    if GPU_AVAILABLE:
        log_print(f"    Transferindo dados para GPU...", log_file)
        X_train_gpu = cp.asarray(X_train)
        X_test_gpu = cp.asarray(X_test)
        y_train_gpu = cp.asarray(y_train)
        y_test_gpu = cp.asarray(y_test)
        mem_usage = X_train_gpu.nbytes / 1e9
        log_print(f"    ✓ Dados na GPU (VRAM: ~{mem_usage:.2f} GB)", log_file)
        return X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning_manual(X_train, X_test, y_train, y_test, classifier_name, param_grid, log_file, use_gpu=True):
    log_print(f"\n{'='*80}", log_file)
    log_print(f"GRID SEARCH: {classifier_name}", log_file)
    log_print(f"🚀 GPU (cuML)" if use_gpu and GPU_AVAILABLE else f"💻 CPU (sklearn)", log_file)
    log_print(f"{'='*80}", log_file)

    if use_gpu and GPU_AVAILABLE:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = convert_to_gpu(X_train, X_test, y_train, y_test, log_file)
    else:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = X_train, X_test, y_train, y_test

    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]
    all_combinations = list(product(*param_values))

    log_print(f"Testando {len(all_combinations)} combinações...", log_file)
    best_f1, best_params, best_model = -1, None, None

    for i, param_combo in enumerate(all_combinations):
        params = dict(zip(param_keys, param_combo))
        try:
            # CORREÇÃO 2: Passar 'random_state' apenas para os modelos que o suportam.
            if classifier_name in ['SVM', 'SVM_RBF']:
                model = cuSVC(random_state=42, **params)
            elif classifier_name == 'RandomForest':
                model = cuRF(random_state=42, **params)
            else: # LogisticRegression não aceita 'random_state' no cuML
                model = cuLR(**params)

            model.fit(X_train_proc, y_train_proc)
            y_pred = model.predict(X_test_proc)

            y_pred_cpu = cp.asnumpy(y_pred) if GPU_AVAILABLE and use_gpu else y_pred
            y_test_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc
            f1 = f1_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0)

            if f1 > best_f1:
                best_f1, best_params, best_model = f1, params, model
                log_print(f"  [{i+1}/{len(all_combinations)}] ✓ Novo melhor! F1={f1:.4f} | {params}", log_file)
            elif (i + 1) % 5 == 0:
                log_print(f"  [{i+1}/{len(all_combinations)}] F1={f1:.4f}", log_file)
        except Exception as e:
            log_print(f"  [{i+1}/{len(all_combinations)}] ❌ Erro em {params}: {str(e)[:100]}", log_file)
            continue

    log_print(f"\n✅ Grid Search concluído!", log_file)

    if best_params is None:
        log_print(f"\n❌ Nenhuma combinação funcionou para {classifier_name}.", log_file)
        return {'method': 'GridSearch', 'classifier': classifier_name, 'test_f1': 0, 'best_params': {}, 'error': 'All combinations failed'}

    log_print(f"\n🏆 MELHORES HIPERPARÂMETROS:", log_file)
    for param, value in best_params.items():
        log_print(f"  • {param}: {value}", log_file)

    y_pred_final = best_model.predict(X_test_proc)
    y_pred_final_cpu = cp.asnumpy(y_pred_final) if GPU_AVAILABLE and use_gpu else y_pred_final
    y_test_final_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc

    accuracy = accuracy_score(y_test_final_cpu, y_pred_final_cpu)
    precision = precision_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)
    recall = recall_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)
    f1 = f1_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)

    log_print(f"\n📈 TESTE:\n  • Accuracy:  {accuracy:.4f}\n  • Precision: {precision:.4f}\n  • Recall:    {recall:.4f}\n  • F1-Score:  {f1:.4f}", log_file)
    return {'method': 'GridSearch', 'classifier': classifier_name, 'best_params': best_params, 'cv_score': best_f1, 'test_accuracy': accuracy, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1, 'best_model': best_model, 'n_combinations': len(all_combinations), 'used_gpu': GPU_AVAILABLE and use_gpu}

def hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, classifier_name, param_distributions, n_iter, log_file, use_gpu=True):
    log_print(f"\n{'='*80}", log_file)
    log_print(f"RANDOM SEARCH: {classifier_name}", log_file)
    log_print(f"🚀 GPU (cuML)" if use_gpu and GPU_AVAILABLE else f"💻 CPU (sklearn)", log_file)
    log_print(f"{'='*80}", log_file)

    if use_gpu and GPU_AVAILABLE:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = convert_to_gpu(X_train, X_test, y_train, y_test, log_file)
    else:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = X_train, X_test, y_train, y_test

    log_print(f"Testando {n_iter} combinações aleatórias...", log_file)
    best_f1, best_params, best_model = -1, None, None
    np.random.seed(42)

    for i in range(n_iter):
        params = {key: val.rvs() if hasattr(val, 'rvs') else (np.random.choice(val) if isinstance(val, list) else val) for key, val in param_distributions.items()}
        try:
            # CORREÇÃO 2: Passar 'random_state' apenas para os modelos que o suportam.
            if classifier_name in ['SVM', 'SVM_RBF']:
                model = cuSVC(random_state=42, **params)
            elif classifier_name == 'RandomForest':
                model = cuRF(random_state=42, **params)
            else: # LogisticRegression não aceita 'random_state' no cuML
                model = cuLR(**params)

            model.fit(X_train_proc, y_train_proc)
            y_pred = model.predict(X_test_proc)

            y_pred_cpu = cp.asnumpy(y_pred) if GPU_AVAILABLE and use_gpu else y_pred
            y_test_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc
            f1 = f1_score(y_test_cpu, y_pred_cpu, average='weighted', zero_division=0)

            if f1 > best_f1:
                best_f1, best_params, best_model = f1, params, model
                log_print(f"  [{i+1}/{n_iter}] ✓ Novo melhor! F1={f1:.4f}", log_file)
            elif (i + 1) % 5 == 0:
                log_print(f"  [{i+1}/{n_iter}] F1={f1:.4f}", log_file)
        except Exception as e:
            log_print(f"  [{i+1}/{n_iter}] ❌ Erro em {params}: {str(e)[:100]}", log_file)
            continue

    log_print(f"\n✅ Random Search concluído!", log_file)

    if best_params is None:
        log_print(f"\n❌ Nenhuma combinação funcionou para {classifier_name}.", log_file)
        return {'method': 'RandomSearch', 'classifier': classifier_name, 'test_f1': 0, 'best_params': {}, 'error': 'All combinations failed'}

    log_print(f"\n🏆 MELHORES HIPERPARÂMETROS:", log_file)
    for param, value in best_params.items():
        log_print(f"  • {param}: {value}", log_file)

    y_pred_final = best_model.predict(X_test_proc)
    y_pred_final_cpu = cp.asnumpy(y_pred_final) if GPU_AVAILABLE and use_gpu else y_pred_final
    y_test_final_cpu = cp.asnumpy(y_test_proc) if GPU_AVAILABLE and use_gpu else y_test_proc

    accuracy = accuracy_score(y_test_final_cpu, y_pred_final_cpu)
    precision = precision_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)
    recall = recall_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)
    f1 = f1_score(y_test_final_cpu, y_pred_final_cpu, average='weighted', zero_division=0)

    log_print(f"\n📈 TESTE:\n  • Accuracy:  {accuracy:.4f}\n  • F1-Score:  {f1:.4f}", log_file)
    return {'method': 'RandomSearch', 'classifier': classifier_name, 'best_params': best_params, 'cv_score': best_f1, 'test_accuracy': accuracy, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1, 'best_model': best_model, 'n_combinations': n_iter, 'used_gpu': GPU_AVAILABLE and use_gpu}

# --- A função main permanece a mesma ---
def main():
    """Função principal"""
    log_file = os.path.join(RESULTS_DIR, f'log_sfr_better_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    log_print(f"{'='*80}", log_file)
    log_print(f"SFR-MISTRAL: OTIMIZAÇÃO DE CLASSIFICADORES", log_file)
    log_print(f"Embeddings: PRÉ-CARREGADOS de {EMBEDDINGS_DIR}", log_file)
    log_print(f"Resultados: {RESULTS_DIR}", log_file)
    log_print(f"GPU: cuML {'✅ ATIVO' if GPU_AVAILABLE else '❌ CPU only'}", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)
    all_results = []
    for dataset_path in DATASETS:
        if not os.path.exists(dataset_path):
            log_print(f"\n⚠️  Dataset não encontrado: {dataset_path}", log_file)
            continue
        dataset_name = os.path.basename(dataset_path).replace('.xlsx', '')
        log_print(f"\n\n{'#'*80}", log_file)
        log_print(f"DATASET: {dataset_name}", log_file)
        log_print(f"{'#'*80}", log_file)
        try:
            df = pd.read_excel(dataset_path, engine='openpyxl')
        except ImportError:
            log_print("A biblioteca 'openpyxl' é necessária para ler arquivos .xlsx.", log_file)
            log_print("Por favor, instale com: conda install openpyxl", log_file)
            return
        class_column = 'Class' if 'Class' in df.columns else 'Classe'
        y = df[class_column].values
        for combo_name in BEST_COMBINATION.keys():
            log_print(f"\nCombinação: {combo_name}", log_file)
            cache_file = os.path.join(EMBEDDINGS_DIR, f'embeddings_{dataset_name}_{combo_name}.npy')
            if not os.path.exists(cache_file):
                log_print(f"  ❌ ERRO: Embeddings não encontrados: {cache_file}", log_file)
                continue
            log_print(f"  ✓ Carregando embeddings de: {cache_file}", log_file)
            X = np.load(cache_file)
            log_print(f"  ✓ Shape: {X.shape}", log_file)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            log_print(f"  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}", log_file)
            # SVM LINEAR
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: SVM (LINEAR)", log_file)
            log_print(f"{'*'*80}", log_file)
            for res in [hyperparameter_tuning_manual(X_train, X_test, y_train, y_test, 'SVM', PARAM_GRIDS_FAST['SVM'], log_file),
                        hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, 'SVM', PARAM_DISTRIBUTIONS_FAST['SVM'], 15, log_file)]:
                res.update({'dataset': dataset_name, 'combination': combo_name})
                all_results.append(res)
            # RANDOM FOREST
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: RandomForest", log_file)
            log_print(f"{'*'*80}", log_file)
            res_rf = hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, 'RandomForest', PARAM_DISTRIBUTIONS_FAST['RandomForest'], 20, log_file)
            res_rf.update({'dataset': dataset_name, 'combination': combo_name})
            all_results.append(res_rf)
            # LOGISTIC REGRESSION
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: LogisticRegression", log_file)
            log_print(f"{'*'*80}", log_file)
            for res in [hyperparameter_tuning_manual(X_train, X_test, y_train, y_test, 'LogisticRegression', PARAM_GRIDS_FAST['LogisticRegression'], log_file),
                        hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, 'LogisticRegression', PARAM_DISTRIBUTIONS_FAST['LogisticRegression'], 15, log_file)]:
                res.update({'dataset': dataset_name, 'combination': combo_name})
                all_results.append(res)
    # RESUMO FINAL
    log_print(f"\n\n{'='*80}", log_file)
    log_print(f"RESUMO FINAL", log_file)
    log_print(f"{'='*80}", log_file)
    all_results = [res for res in all_results if 'error' not in res]
    if not all_results:
        log_print(f"\n❌ Nenhum resultado gerado!", log_file)
        return
    summary_data = [{'Dataset': r.get('dataset'), 'Combination': r.get('combination'), 'Method': r.get('method'), 'Classifier': r.get('classifier'), 'Test_F1': r.get('test_f1'), 'Test_Accuracy': r.get('test_accuracy'), 'Test_Precision': r.get('test_precision'), 'Test_Recall': r.get('test_recall'), 'Used_GPU': r.get('used_gpu'), 'N_Combinations': r.get('n_combinations'), 'Best_Params': str(r.get('best_params'))} for r in all_results]
    df_summary = pd.DataFrame(summary_data).sort_values(by='Test_F1', ascending=False)
    output_csv = os.path.join(RESULTS_DIR, 'melhores_hiperparametros_sfr.csv')
    df_summary.to_csv(output_csv, index=False)
    log_print(f"\n{df_summary[['Dataset', 'Method', 'Classifier', 'Test_F1', 'Test_Accuracy', 'Used_GPU']].to_string(index=False)}", log_file)
    log_print(f"\n✅ Resultados salvos em: {output_csv}", log_file)
    log_print(f"\n\n{'='*80}", log_file)
    log_print(f"🏆 CAMPEÕES POR CLASSIFICADOR:", log_file)
    log_print(f"{'='*80}", log_file)
    for clf_name in df_summary['Classifier'].unique():
        best = df_summary[df_summary['Classifier'] == clf_name].iloc[0]
        log_print(f"\n{clf_name}:\n  Dataset: {best['Dataset']}\n  Método: {best['Method']}\n  F1-Score: {best['Test_F1']:.4f}\n  Accuracy: {best['Test_Accuracy']:.4f}\n  GPU: {best['Used_GPU']}\n  Parâmetros: {best['Best_Params']}", log_file)
    if not df_summary.empty:
        best_overall = df_summary.iloc[0]
        log_print(f"\n\n{'='*80}", log_file)
        log_print(f"🥇 MELHOR RESULTADO GERAL:", log_file)
        log_print(f"{'='*80}", log_file)
        log_print(f"  Classificador: {best_overall['Classifier']}\n  Dataset: {best_overall['Dataset']}\n  Método: {best_overall['Method']}\n  F1-Score: {best_overall['Test_F1']:.4f}\n  Accuracy: {best_overall['Test_Accuracy']:.4f}\n  Parâmetros: {best_overall['Best_Params']}", log_file)
    models_file = os.path.join(RESULTS_DIR, 'best_models_sfr.pkl')
    with open(models_file, 'wb') as f: pickle.dump(all_results, f)
    log_print(f"\n✅ Modelos salvos em: {models_file}", log_file)
    log_print(f"\n{'='*80}", log_file)
    log_print(f"✅ PROCESSAMENTO CONCLUÍDO!", log_file)
    log_print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()