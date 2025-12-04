import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint
import warnings
import cuml  # RAPIDS para GPU (opcional)
try:
    from cuml.svm import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    GPU_AVAILABLE = True
    print("✅ RAPIDS cuML disponível - Usando GPU!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  RAPIDS não instalado - Usando CPU (mais lento)")
    print("   Para instalar: conda install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11 cuda-version=12.0")

warnings.filterwarnings('ignore')

# Configurações
DATASETS = [
    '../../../bases_preprocessados/dataset_sem_stopwords_lemmatizado.xlsx',
    '../../../bases_preprocessados/dataset_sem_caracteres_especiais_lemmatizado.xlsx'
]

BEST_COMBINATION = {
    'completo': ['Titulo', 'Subtitulo', 'Noticia']
}

# GRIDS OTIMIZADOS (REDUZIDOS) - Focado em velocidade
PARAM_GRIDS_FAST = {
    'SVM': {
        'C': [0.1, 1.0, 10.0, 100.0],  # 4 valores (era 9)
        'kernel': ['linear'],  # Apenas linear (mais rápido, melhor para TF-IDF)
        'class_weight': [None, 'balanced']  # 2 valores
        # Total: 4 × 1 × 2 = 8 combinações (5-fold CV = 40 fits)
    },

    'SVM_RBF': {  # Grid separado para RBF (opcional, mais lento)
        'C': [1.0, 10.0],  # Apenas 2 valores
        'kernel': ['rbf'],
        'gamma': ['scale', 0.01],  # Apenas 2 gammas
        'class_weight': ['balanced']  # Apenas balanced
        # Total: 2 × 1 × 2 × 1 = 4 combinações (5-fold CV = 20 fits)
    },

    'RandomForest': {
        'n_estimators': [100, 200, 300],  # 3 valores (era 5)
        'max_depth': [20, 30, None],  # 3 valores (era 5)
        'min_samples_split': [2, 5, 10],  # 3 valores (era 4)
        'min_samples_leaf': [1, 2, 4],  # 3 valores (era 4)
        'max_features': ['sqrt', 'log2'],  # 2 valores (era 4)
        'class_weight': [None, 'balanced'],  # 2 valores (era 3)
        'criterion': ['gini']  # Apenas gini (entropy é mais lento)
        # Total: 3×3×3×3×2×2×1 = 324 combinações
        # Vamos usar RandomSearch com n_iter=50 ao invés
    },

    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0, 100.0],  # 4 valores (era 9)
        'penalty': ['l2'],  # Apenas l2 (mais estável, não precisa saga)
        'solver': ['lbfgs'],  # lbfgs é mais rápido que saga para l2
        'class_weight': [None, 'balanced'],  # 2 valores
        'max_iter': [1000]  # Fixo (geralmente converge rápido)
        # Total: 4 × 1 × 1 × 2 × 1 = 8 combinações (5-fold CV = 40 fits)
    }
}

# Random Search - Mais eficiente
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
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.5],
        'class_weight': [None, 'balanced'],
        'criterion': ['gini']
    },

    'LogisticRegression': {
        'C': uniform(0.1, 100),
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000]
    }
}

RESULTS_DIR = 'results_tfidf_hypertuning_fast'
os.makedirs(RESULTS_DIR, exist_ok=True)

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
                combined.append(str(row[field_name]))
        texts.append(' '.join(combined))

    return texts

def hyperparameter_tuning_grid(X_train, X_test, y_train, y_test, classifier_name, param_grid, log_file):
    """Grid Search OTIMIZADO"""
    log_print(f"\n{'='*80}", log_file)
    log_print(f"GRID SEARCH (OTIMIZADO): {classifier_name}", log_file)
    log_print(f"{'='*80}", log_file)

    # Calcular número de combinações
    num_combinations = np.prod([len(v) for v in param_grid.values()])
    log_print(f"Combinações a testar: {int(num_combinations)} (com 5-fold CV = {int(num_combinations * 5)} fits)", log_file)

    # Criar classificador base
    if classifier_name == 'SVM' or classifier_name == 'SVM_RBF':
        base_estimator = SVC(random_state=42, cache_size=2000)  # Cache maior
    elif classifier_name == 'RandomForest':
        base_estimator = RandomForestClassifier(random_state=42, n_jobs=-1, warm_start=False)
    else:  # LogisticRegression
        base_estimator = LogisticRegression(random_state=42, n_jobs=-1, warm_start=False)

    # Grid Search com 3-fold (ao invés de 5) para ser mais rápido
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=3,  # REDUZIDO: 3-fold ao invés de 5
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        return_train_score=False  # Não retorna train score (economiza memória)
    )

    log_print(f"Iniciando Grid Search (3-fold CV)...", log_file)
    grid_search.fit(X_train, y_train)

    log_print(f"\n✅ Grid Search concluído!", log_file)
    log_print(f"\n🏆 MELHORES HIPERPARÂMETROS:", log_file)
    for param, value in grid_search.best_params_.items():
        log_print(f"  • {param}: {value}", log_file)

    log_print(f"\n📊 Score de validação cruzada (CV): {grid_search.best_score_:.4f}", log_file)

    # Avaliar no conjunto de teste
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    log_print(f"\n📈 PERFORMANCE NO TESTE:", log_file)
    log_print(f"  • Accuracy:  {accuracy:.4f}", log_file)
    log_print(f"  • Precision: {precision:.4f}", log_file)
    log_print(f"  • Recall:    {recall:.4f}", log_file)
    log_print(f"  • F1-Score:  {f1:.4f}", log_file)

    return {
        'method': 'GridSearch',
        'classifier': classifier_name,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'best_model': best_model,
        'n_combinations': int(num_combinations)
    }

def hyperparameter_tuning_random(X_train, X_test, y_train, y_test, classifier_name, param_distributions, n_iter, log_file):
    """Random Search OTIMIZADO"""
    log_print(f"\n{'='*80}", log_file)
    log_print(f"RANDOM SEARCH (OTIMIZADO): {classifier_name}", log_file)
    log_print(f"{'='*80}", log_file)

    # Criar classificador base
    if classifier_name == 'SVM':
        base_estimator = SVC(random_state=42, cache_size=2000)
    elif classifier_name == 'RandomForest':
        base_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
    else:  # LogisticRegression
        base_estimator = LogisticRegression(random_state=42, n_jobs=-1)

    log_print(f"Testando {n_iter} combinações aleatórias (3-fold CV)...", log_file)

    # Random Search com 3-fold
    random_search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,  # REDUZIDO: 3-fold
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=False
    )

    random_search.fit(X_train, y_train)

    log_print(f"\n✅ Random Search concluído!", log_file)
    log_print(f"\n🏆 MELHORES HIPERPARÂMETROS:", log_file)
    for param, value in random_search.best_params_.items():
        log_print(f"  • {param}: {value}", log_file)

    log_print(f"\n📊 Score de validação cruzada (CV): {random_search.best_score_:.4f}", log_file)

    # Avaliar no teste
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    log_print(f"\n📈 PERFORMANCE NO TESTE:", log_file)
    log_print(f"  • Accuracy:  {accuracy:.4f}", log_file)
    log_print(f"  • Precision: {precision:.4f}", log_file)
    log_print(f"  • Recall:    {recall:.4f}", log_file)
    log_print(f"  • F1-Score:  {f1:.4f}", log_file)

    return {
        'method': 'RandomSearch',
        'classifier': classifier_name,
        'best_params': random_search.best_params_,
        'cv_score': random_search.best_score_,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'best_model': best_model,
        'n_combinations': n_iter
    }

def main():
    """Função principal"""
    log_file = os.path.join(RESULTS_DIR, f'log_tfidf_hypertuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    log_print(f"{'='*80}", log_file)
    log_print(f"TF-IDF: HYPERPARAMETER TUNING (VERSÃO OTIMIZADA/RÁPIDA)", log_file)
    log_print(f"Otimizações:", log_file)
    log_print(f"  • 3-fold CV (ao invés de 5-fold)", log_file)
    log_print(f"  • SVM: apenas kernel linear + grid reduzido para RBF", log_file)
    log_print(f"  • RandomForest: Random Search ao invés de Grid", log_file)
    log_print(f"  • LogisticRegression: apenas L2 com lbfgs", log_file)
    log_print(f"  • Grids reduzidos (~200 testes totais)", log_file)
    log_print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

    all_results = []

    # Processar cada dataset
    for dataset_path in DATASETS:
        if not os.path.exists(dataset_path):
            log_print(f"\nAVISO: {dataset_path} não encontrado!", log_file)
            continue

        dataset_name = os.path.basename(dataset_path).replace('.xlsx', '')

        log_print(f"\n\n{'#'*80}", log_file)
        log_print(f"DATASET: {dataset_name}", log_file)
        log_print(f"{'#'*80}", log_file)

        # Carregar dados
        df = pd.read_excel(dataset_path)
        class_column = 'Class' if 'Class' in df.columns else 'Classe'

        # Processar melhor combinação
        for combo_name, fields in BEST_COMBINATION.items():
            log_print(f"\nCombinação: {combo_name} - {fields}", log_file)

            # Gerar embeddings TF-IDF
            texts = combine_fields(df, fields, use_processed=True)
            y = df[class_column].astype(str).to_numpy()

            log_print(f"\nGerando embeddings TF-IDF otimizados...", log_file)
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            X = vectorizer.fit_transform(texts)
            log_print(f"Dimensão TF-IDF: {X.shape}", log_file)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # ==========================================
            # SVM LINEAR (GRID SEARCH - RÁPIDO)
            # ==========================================
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: SVM (LINEAR)", log_file)
            log_print(f"{'*'*80}", log_file)

            grid_results_svm = hyperparameter_tuning_grid(
                X_train, X_test, y_train, y_test,
                'SVM', PARAM_GRIDS_FAST['SVM'], log_file
            )
            grid_results_svm['dataset'] = dataset_name
            grid_results_svm['combination'] = combo_name
            all_results.append(grid_results_svm)

            # Random Search SVM Linear
            random_results_svm = hyperparameter_tuning_random(
                X_train, X_test, y_train, y_test,
                'SVM', PARAM_DISTRIBUTIONS_FAST['SVM'],
                n_iter=20,  # Apenas 20 iterações
                log_file=log_file
            )
            random_results_svm['dataset'] = dataset_name
            random_results_svm['combination'] = combo_name
            all_results.append(random_results_svm)

            # ==========================================
            # SVM RBF (OPCIONAL - GRID PEQUENO)
            # ==========================================
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: SVM (RBF - OPCIONAL)", log_file)
            log_print(f"{'*'*80}", log_file)

            grid_results_svm_rbf = hyperparameter_tuning_grid(
                X_train, X_test, y_train, y_test,
                'SVM_RBF', PARAM_GRIDS_FAST['SVM_RBF'], log_file
            )
            grid_results_svm_rbf['dataset'] = dataset_name
            grid_results_svm_rbf['combination'] = combo_name
            all_results.append(grid_results_svm_rbf)

            # ==========================================
            # RANDOM FOREST (RANDOM SEARCH - RÁPIDO)
            # ==========================================
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: RandomForest", log_file)
            log_print(f"{'*'*80}", log_file)

            # Apenas Random Search (Grid seria muito lento)
            random_results_rf = hyperparameter_tuning_random(
                X_train, X_test, y_train, y_test,
                'RandomForest', PARAM_DISTRIBUTIONS_FAST['RandomForest'],
                n_iter=50,  # 50 iterações
                log_file=log_file
            )
            random_results_rf['dataset'] = dataset_name
            random_results_rf['combination'] = combo_name
            all_results.append(random_results_rf)

            # ==========================================
            # LOGISTIC REGRESSION (GRID SEARCH - RÁPIDO)
            # ==========================================
            log_print(f"\n\n{'*'*80}", log_file)
            log_print(f"CLASSIFICADOR: LogisticRegression", log_file)
            log_print(f"{'*'*80}", log_file)

            grid_results_lr = hyperparameter_tuning_grid(
                X_train, X_test, y_train, y_test,
                'LogisticRegression', PARAM_GRIDS_FAST['LogisticRegression'], log_file
            )
            grid_results_lr['dataset'] = dataset_name
            grid_results_lr['combination'] = combo_name
            all_results.append(grid_results_lr)

            # Random Search LR
            random_results_lr = hyperparameter_tuning_random(
                X_train, X_test, y_train, y_test,
                'LogisticRegression', PARAM_DISTRIBUTIONS_FAST['LogisticRegression'],
                n_iter=20,  # 20 iterações
                log_file=log_file
            )
            random_results_lr['dataset'] = dataset_name
            random_results_lr['combination'] = combo_name
            all_results.append(random_results_lr)

    # Salvar resumo final
    log_print(f"\n\n{'='*80}", log_file)
    log_print(f"RESUMO FINAL - MELHORES HIPERPARÂMETROS", log_file)
    log_print(f"{'='*80}", log_file)

    summary_data = []
    total_combinations = 0
    for result in all_results:
        summary_data.append({
            'Dataset': result['dataset'],
            'Combination': result['combination'],
            'Method': result['method'],
            'Classifier': result['classifier'],
            'CV_Score': result['cv_score'],
            'Test_F1': result['test_f1'],
            'Test_Accuracy': result['test_accuracy'],
            'N_Combinations': result['n_combinations'],
            'Best_Params': str(result['best_params'])
        })
        total_combinations += result['n_combinations']

    df_summary = pd.DataFrame(summary_data)
    output_csv = os.path.join(RESULTS_DIR, 'melhores_hiperparametros_tfidf.csv')
    df_summary.to_csv(output_csv, index=False)

    log_print(f"\n{df_summary.to_string(index=False)}", log_file)
    log_print(f"\n✅ Resultados salvos em: {output_csv}", log_file)
    log_print(f"\n📊 Total de combinações testadas: {total_combinations}", log_file)

    # Melhor de cada classificador
    log_print(f"\n\n{'='*80}", log_file)
    log_print(f"🏆 CAMPEÕES POR CLASSIFICADOR:", log_file)
    log_print(f"{'='*80}", log_file)

    for clf_name in ['SVM', 'SVM_RBF', 'RandomForest', 'LogisticRegression']:
        clf_results = df_summary[df_summary['Classifier'] == clf_name]
        if len(clf_results) > 0:
            best = clf_results.loc[clf_results['Test_F1'].idxmax()]
            log_print(f"\n{clf_name}:", log_file)
            log_print(f"  Método: {best['Method']}", log_file)
            log_print(f"  F1-Score: {best['Test_F1']:.4f}", log_file)
            log_print(f"  CV Score: {best['CV_Score']:.4f}", log_file)
            log_print(f"  Parâmetros: {best['Best_Params']}", log_file)

    # Salvar modelos
    models_file = os.path.join(RESULTS_DIR, 'best_models_tfidf.pkl')
    with open(models_file, 'wb') as f:
        pickle.dump(all_results, f)
    log_print(f"\n✅ Modelos salvos em: {models_file}", log_file)

    log_print(f"\n{'='*80}", log_file)
    log_print(f"PROCESSAMENTO CONCLUÍDO!", log_file)
    log_print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"{'='*80}", log_file)

if __name__ == "__main__":
    main()
