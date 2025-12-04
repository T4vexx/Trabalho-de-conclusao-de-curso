import pandas as pd
import numpy as np
import pickle
import os
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Tentar importar cuML (GPU) - se falhar, usa sklearn (CPU)
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
    print("   Para instalar GPU:")
    print("   conda install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11 cuda-version=12.0")

# Configurações
DATASETS = [
    '../../../bases_preprocessados/dataset_sem_stopwords_lemmatizado.xlsx',
    '../../../bases_preprocessados/dataset_sem_caracteres_especiais_lemmatizado.xlsx'
]

BEST_COMBINATION = {
    'completo': ['Titulo', 'Subtitulo', 'Noticia']
}

# Modelo SFR
SFR_MODEL = 'Salesforce/SFR-Embedding-Mistral'
BATCH_SIZE = 32
EMBEDDING_DIM = 4096
MAX_SEQ_LENGTH = 256


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
        'solver': ['lbfgs'] if not GPU_AVAILABLE else ['qn'],  # qn para cuML
        'class_weight': [None, 'balanced'],
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
        'max_depth': [20, 30, 50, None],
        'min_samples_split': randint(2, 11),
        'max_features': ['sqrt', 'log2']
    },

    'LogisticRegression': {
        'C': uniform(0.1, 100),
        'penalty': ['l2'],
        'solver': ['qn'] if GPU_AVAILABLE else ['lbfgs'],
        'class_weight': [None, 'balanced'],
        'max_iter': [2000]
    }
}

EMBEDDINGS_DIR = 'embeddings_sfr'
RESULTS_DIR = 'results_sfr_hypertuning_gpu'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def log_print(message, log_file):
    """Imprime e salva no log"""
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

def add_sfr_instruction(texts):
    """Adiciona instrução SFR"""
    instruction = "Represent this sentence for searching relevant passages: "
    return [instruction + text for text in texts]

def get_sfr_embeddings(texts, model, device, log_file):
    """Gera embeddings SFR-Mistral"""
    log_print(f"    Gerando {len(texts)} embeddings SFR-Mistral...", log_file)

    texts_with_instruction = add_sfr_instruction(texts)

    if device == 'cuda':
        before_mem = torch.cuda.memory_allocated(0) / 1e9
        log_print(f"    VRAM antes: {before_mem:.2f} GB", log_file)

    embeddings = model.encode(
        texts_with_instruction,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )

    if device == 'cuda':
        after_mem = torch.cuda.memory_allocated(0) / 1e9
        log_print(f"    VRAM depois: {after_mem:.2f} GB", log_file)
        torch.cuda.empty_cache()

    log_print(f"    ✓ Shape: {embeddings.shape}", log_file)

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    return embeddings

def convert_to_gpu(X_train, X_test, y_train, y_test, log_file):
    """Converte dados para GPU se cuML disponível"""
    if GPU_AVAILABLE:
        log_print(f"    Transferindo dados para GPU...", log_file)
        import cupy as cp
        X_train_gpu = cp.asarray(X_train)
        X_test_gpu = cp.asarray(X_test)
        y_train_gpu = cp.asarray(y_train)
        y_test_gpu = cp.asarray(y_test)
        log_print(f"    ✓ Dados na GPU (VRAM: ~{X_train_gpu.nbytes / 1e9:.2f} GB)", log_file)
        return X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu
    else:
        return X_train, X_test, y_train, y_test

def hyperparameter_tuning_manual(X_train, X_test, y_train, y_test, classifier_name, param_grid, log_file, use_gpu=True):
    """
    Grid Search MANUAL (cuML não tem GridSearchCV ainda)
    Testa todas as combinações manualmente
    """
    log_print(f"\n{'='*80}", log_file)
    log_print(f"GRID SEARCH MANUAL: {classifier_name}", log_file)
    if use_gpu and GPU_AVAILABLE:
        log_print(f"🚀 RODANDO NA GPU (cuML)", log_file)
    else:
        log_print(f"💻 Rodando na CPU (sklearn)", log_file)
    log_print(f"{'='*80}", log_file)

    # Converter para GPU se disponível
    if use_gpu and GPU_AVAILABLE:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = convert_to_gpu(
            X_train, X_test, y_train, y_test, log_file
        )
    else:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = X_train, X_test, y_train, y_test

    # Gerar todas as combinações de parâmetros
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]
    all_combinations = list(product(*param_values))

    log_print(f"Testando {len(all_combinations)} combinações...", log_file)

    best_f1 = -1
    best_params = None
    best_model = None

    for i, param_combo in enumerate(all_combinations):
        params = dict(zip(param_keys, param_combo))

        # Criar modelo com parâmetros específicos
        try:
            if classifier_name == 'SVM' or classifier_name == 'SVM_RBF':
                model = cuSVC(random_state=42, **params)
            elif classifier_name == 'RandomForest':
                model = cuRF(random_state=42, **params)
            else:  # LogisticRegression
                model = cuLR(random_state=42, **params)

            # Treinar
            model.fit(X_train_proc, y_train_proc)

            # Predizer
            y_pred = model.predict(X_test_proc)

            # Converter de volta para CPU se necessário
            if GPU_AVAILABLE and use_gpu:
                import cupy as cp
                y_pred = cp.asnumpy(y_pred)
                y_test_eval = cp.asnumpy(y_test_proc)
            else:
                y_test_eval = y_test_proc

            # Calcular F1
            f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = model
                log_print(f"  [{i+1}/{len(all_combinations)}] ✓ Novo melhor! F1={f1:.4f} | {params}", log_file)
            else:
                if (i + 1) % 5 == 0:
                    log_print(f"  [{i+1}/{len(all_combinations)}] F1={f1:.4f}", log_file)

        except Exception as e:
            log_print(f"  [{i+1}/{len(all_combinations)}] ❌ Erro: {str(e)[:50]}", log_file)
            continue

    log_print(f"\n✅ Grid Search concluído!", log_file)
    log_print(f"\n🏆 MELHORES HIPERPARÂMETROS:", log_file)
    for param, value in best_params.items():
        log_print(f"  • {param}: {value}", log_file)

    # Avaliar melhor modelo no teste completo
    y_pred = best_model.predict(X_test_proc)

    if GPU_AVAILABLE and use_gpu:
        import cupy as cp
        y_pred = cp.asnumpy(y_pred)
        y_test_final = cp.asnumpy(y_test_proc)
    else:
        y_test_final = y_test_proc

    accuracy = accuracy_score(y_test_final, y_pred)
    precision = precision_score(y_test_final, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_final, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_final, y_pred, average='weighted', zero_division=0)

    log_print(f"\n📈 TESTE:", log_file)
    log_print(f"  • Accuracy:  {accuracy:.4f}", log_file)
    log_print(f"  • Precision: {precision:.4f}", log_file)
    log_print(f"  • Recall:    {recall:.4f}", log_file)
    log_print(f"  • F1-Score:  {f1:.4f}", log_file)

    return {
        'method': 'GridSearch_Manual',
        'classifier': classifier_name,
        'best_params': best_params,
        'cv_score': best_f1,  # Não temos CV, mas temos F1 de teste
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'best_model': best_model,
        'n_combinations': len(all_combinations),
        'used_gpu': GPU_AVAILABLE and use_gpu
    }

def hyperparameter_tuning_random_manual(X_train, X_test, y_train, y_test, classifier_name, param_distributions, n_iter, log_file, use_gpu=True):
    """Random Search MANUAL com GPU"""
    log_print(f"\n{'='*80}", log_file)
    log_print(f"RANDOM SEARCH MANUAL: {classifier_name}", log_file)
    if use_gpu and GPU_AVAILABLE:
        log_print(f"🚀 RODANDO NA GPU (cuML)", log_file)
    else:
        log_print(f"💻 Rodando na CPU (sklearn)", log_file)
    log_print(f"{'='*80}", log_file)

    # Converter para GPU se disponível
    if use_gpu and GPU_AVAILABLE:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = convert_to_gpu(
            X_train, X_test, y_train, y_test, log_file
        )
    else:
        X_train_proc, X_test_proc, y_train_proc, y_test_proc = X_train, X_test, y_train, y_test

    log_print(f"Testando {n_iter} combinações aleatórias...", log_file)

    best_f1 = -1
    best_params = None
    best_model = None

    np.random.seed(42)

    for i in range(n_iter):
        # Gerar parâmetros aleatórios
        params = {}
        for key, value in param_distributions.items():
            if isinstance(value, list):
                params[key] = np.random.choice(value)
            elif hasattr(value, 'rvs'):  # scipy.stats distribution
                params[key] = value.rvs()
            else:
                params[key] = value

        try:
            if classifier_name == 'SVM' or classifier_name == 'SVM_RBF':
                model = cuSVC(random_state=42, **params)
            elif classifier_name == 'RandomForest':
                model = cuRF(random_state=42, **params)
            else:
                model = cuLR(random_state=42, **params)

            model.fit(X_train_proc, y_train_proc)
            y_pred = model.predict(X_test_proc)

            if GPU_AVAILABLE and use_gpu:
                import cupy as cp
                y_pred = cp.asnumpy(y_pred)
                y_test_eval = cp.asnumpy(y_test_proc)
            else:
                y_test_eval = y_test_proc

            f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = model
                log_print(f"  [{i+1}/{n_iter}] ✓ Novo melhor! F1={f1:.4f}", log_file)
            else:
                if (i + 1) % 5 == 0:
                    log_print(f"  [{i+1}/{n_iter}] F1={f1:.4f}", log_file)

        except Exception as e:
            log_print(f"  [{i+1}/{n_iter}] ❌ Erro: {str(e)[:50]}", log_file)
            continue

    log_print(f"\n✅ Random Search concluído!", log_file)
    log_print(f"\n🏆 MELHORES HIPERPARÂMETROS:", log_file)
    for param, value in best_params.items():
        log_print(f"  • {param}: {value}", log_file)

    # Avaliar
    y_pred = best_model.predict(X_test_proc)

    if GPU_AVAILABLE and use_gpu:
        import cupy as cp
        y_pred = cp.asnumpy(y_pred)
        y_test_final = cp.asnumpy(y_test_proc)
    else:
        y_test_final = y_test_proc

    accuracy = accuracy_score(y_test_final, y_pred)
    precision = precision_score(y_test_final, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_final, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_final, y_pred, average='weighted', zero_division=0)

    log_print(f"\n📈 TESTE:", log_file)
    log_print(f"  • Accuracy:  {accuracy:.4f}", log_file)
    log_print(f"  • F1-Score:  {f1:.4f}", log_file)

    return {
        'method': 'RandomSearch_Manual',
        'classifier': classifier_name,
        'best_params': best_params,
        'cv_score': best_f1,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'best_model': best_model,
        'n_combinations': n_iter,
        'used_gpu': GPU_AVAILABLE and use_gpu
    }

def main():
    """Função principal"""
    log_file = os.path.join(RESULTS_DIR, f'log_sfr_hypertuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    log_print(f"{'='*80}", log_file)
    log_print(f"SFR-MISTRAL: HYPERPARAMETER TUNING COM GPU", log_file)
    log_print(f"{'='*80}", log_file)
    log_print(f"GPU Status:", log_file)
    log_print(f"  • cuML (classificadores): {'✅ ATIVO' if GPU_AVAILABLE else '❌ CPU only'}", log_file)
    log_print(f"  • PyTorch (embeddings): {'✅ CUDA' if torch.cuda.is_available() else '❌ CPU'}", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

    # Verificar GPU PyTorch
    if not torch.cuda.is_available():
        log_print(f"\n⚠️  PyTorch CUDA não disponível (embeddings em CPU)", log_file)
        device = 'cpu'
    else:
        device = 'cuda'
        log_print(f"\n✅ GPU PyTorch: {torch.cuda.get_device_name(0)}", log_file)
        torch.cuda.empty_cache()

    # Carregar modelo SFR
    log_print(f"\nCarregando SFR-Mistral...", log_file)
    model = SentenceTransformer(SFR_MODEL, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    log_print(f"✓ Modelo carregado", log_file)

    all_results = []

    for dataset_path in DATASETS:
        if not os.path.exists(dataset_path):
            continue

        dataset_name = os.path.basename(dataset_path).replace('.xlsx', '')

        log_print(f"\n\n{'#'*80}", log_file)
        log_print(f"DATASET: {dataset_name}", log_file)
        log_print(f"{'#'*80}", log_file)

        df = pd.read_excel(dataset_path)
        class_column = 'Class' if 'Class' in df.columns else 'Classe'

        for combo_name, fields in BEST_COMBINATION.items():
            log_print(f"\nCombinação: {combo_name}", log_file)

            # Cache de embeddings
            cache_file = os.path.join(EMBEDDINGS_DIR, f'embeddings_{dataset_name}_{combo_name}.npy')

            if os.path.exists(cache_file):
                log_print(f"  ✓ Carregando embeddings do cache...", log_file)
                X = np.load(cache_file)
            else:
                texts = combine_fields(df, fields, use_processed=True)
                X = get_sfr_embeddings(texts, model, device, log_file)
                np.save(cache_file, X)
                log_print(f"  ✓ Embeddings em cache", log_file)

            y = df[class_column].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            log_print(f"\nTreino: {X_train.shape[0]} | Teste: {X_test.shape[0]}", log_file)

            # SVM LINEAR
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: SVM (LINEAR)", log_file)
            log_print(f"{'*'*80}", log_file)

            grid_svm = hyperparameter_tuning_manual(
                X_train, X_test, y_train, y_test,
                'SVM', PARAM_GRIDS_FAST['SVM'], log_file, use_gpu=True
            )
            grid_svm['dataset'] = dataset_name
            grid_svm['combination'] = combo_name
            all_results.append(grid_svm)

            random_svm = hyperparameter_tuning_random_manual(
                X_train, X_test, y_train, y_test,
                'SVM', PARAM_DISTRIBUTIONS_FAST['SVM'], 15, log_file, use_gpu=True
            )
            random_svm['dataset'] = dataset_name
            random_svm['combination'] = combo_name
            all_results.append(random_svm)

            # RANDOM FOREST
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: RandomForest", log_file)
            log_print(f"{'*'*80}", log_file)

            random_rf = hyperparameter_tuning_random_manual(
                X_train, X_test, y_train, y_test,
                'RandomForest', PARAM_DISTRIBUTIONS_FAST['RandomForest'], 20, log_file, use_gpu=True
            )
            random_rf['dataset'] = dataset_name
            random_rf['combination'] = combo_name
            all_results.append(random_rf)

            # LOGISTIC REGRESSION
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: LogisticRegression", log_file)
            log_print(f"{'*'*80}", log_file)

            grid_lr = hyperparameter_tuning_manual(
                X_train, X_test, y_train, y_test,
                'LogisticRegression', PARAM_GRIDS_FAST['LogisticRegression'], log_file, use_gpu=True
            )
            grid_lr['dataset'] = dataset_name
            grid_lr['combination'] = combo_name
            all_results.append(grid_lr)

            random_lr = hyperparameter_tuning_random_manual(
                X_train, X_test, y_train, y_test,
                'LogisticRegression', PARAM_DISTRIBUTIONS_FAST['LogisticRegression'], 15, log_file, use_gpu=True
            )
            random_lr['dataset'] = dataset_name
            random_lr['combination'] = combo_name
            all_results.append(random_lr)

            if device == 'cuda':
                torch.cuda.empty_cache()

    # Resumo
    log_print(f"\n\n{'='*80}", log_file)
    log_print(f"RESUMO FINAL", log_file)
    log_print(f"{'='*80}", log_file)

    summary_data = []
    for result in all_results:
        summary_data.append({
            'Dataset': result['dataset'],
            'Method': result['method'],
            'Classifier': result['classifier'],
            'Test_F1': result['test_f1'],
            'Test_Accuracy': result['test_accuracy'],
            'Used_GPU': result['used_gpu'],
            'N_Combinations': result['n_combinations'],
            'Best_Params': str(result['best_params'])
        })

    df_summary = pd.DataFrame(summary_data)
    output_csv = os.path.join(RESULTS_DIR, 'melhores_hiperparametros_sfr_gpu.csv')
    df_summary.to_csv(output_csv, index=False)

    log_print(f"\n{df_summary.to_string(index=False)}", log_file)
    log_print(f"\n✅ Resultados: {output_csv}", log_file)

    # Campeões
    log_print(f"\n\n🏆 CAMPEÕES:", log_file)
    for clf_name in ['SVM', 'RandomForest', 'LogisticRegression']:
        clf_results = df_summary[df_summary['Classifier'] == clf_name]
        if len(clf_results) > 0:
            best = clf_results.loc[clf_results['Test_F1'].idxmax()]
            log_print(f"\n{clf_name}: F1={best['Test_F1']:.4f} (GPU: {best['Used_GPU']})", log_file)
            log_print(f"  {best['Best_Params']}", log_file)

    # Salvar modelos
    models_file = os.path.join(RESULTS_DIR, 'best_models_sfr_gpu.pkl')
    with open(models_file, 'wb') as f:
        pickle.dump(all_results, f)
    log_print(f"\n✅ Modelos salvos: {models_file}", log_file)

    log_print(f"\n{'='*80}", log_file)
    log_print(f"CONCLUÍDO! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()
