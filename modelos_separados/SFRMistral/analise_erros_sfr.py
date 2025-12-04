import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# CONFIGURAÇÕES
# ========================================

RESULTS_DIR = 'results_sfr'
EMBEDDINGS_DIR = 'embeddings_sfr'
ANALYSIS_DIR = 'analise_erros_sfr'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Dataset e combinação que teve melhor resultado
BEST_DATASET = 'dataset_sem_caracteres_especiais_lemmatizado'  # Ajuste conforme seu melhor resultado
BEST_COMBINATION = 'completo'
BEST_CLASSIFIER = 'SVM'

# ========================================
# CARREGAR DADOS SALVOS
# ========================================

def carregar_resultados_completos():
    """Carrega os resultados salvos em pickle"""
    results_pkl = os.path.join(RESULTS_DIR, 'resultados_completos_sfr.pkl')
    
    if not os.path.exists(results_pkl):
        print(f"❌ Arquivo {results_pkl} não encontrado!")
        print(f"Execute primeiro o script de treinamento.")
        return None
    
    with open(results_pkl, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"✅ Resultados carregados: {len(all_results)} experimentos")
    return all_results


def encontrar_melhor_resultado(all_results):
    """Encontra o experimento com melhor F1-Score"""
    best_result = None
    best_f1 = 0
    
    for result in all_results:
        if result['f1_score'] > best_f1:
            best_f1 = result['f1_score']
            best_result = result
    
    print(f"\n{'='*80}")
    print(f"MELHOR RESULTADO ENCONTRADO:")
    print(f"{'='*80}")
    print(f"Dataset: {best_result['dataset']}")
    print(f"Combinação: {best_result['combination']}")
    print(f"Classificador: {best_result['classifier']}")
    print(f"F1-Score: {best_result['f1_score']:.6f}")
    print(f"Accuracy: {best_result['accuracy']:.6f}")
    print(f"{'='*80}\n")
    
    return best_result


def gerar_matriz_confusao(y_test, y_pred, output_dir):
    """Gera e salva a matriz de confusão"""
    cm = confusion_matrix(y_test, y_pred)
    
    # Calcular valores
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*80}")
    print(f"MATRIZ DE CONFUSÃO")
    print(f"{'='*80}")
    print(f"Verdadeiros Negativos (VN): {tn}")
    print(f"Falsos Positivos (FP): {fp}")
    print(f"Falsos Negativos (FN): {fn}")
    print(f"Verdadeiros Positivos (VP): {tp}")
    print(f"{'='*80}\n")
    
    # Criar visualização
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Verdadeira', 'Falsa'],
                yticklabels=['Verdadeira', 'Falsa'])
    plt.title('Matriz de Confusão - SFR-Embedding-Mistral')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'matriz_confusao_sfr.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Matriz de confusão salva em: {output_file}")
    plt.close()
    
    return cm, {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def extrair_exemplos_erros(dataset_path, y_test_indices, y_test, y_pred, output_dir, n_exemplos=10):
    """Extrai exemplos de erros do dataset original"""
    
    print(f"\n{'='*80}")
    print(f"EXTRAINDO EXEMPLOS DE ERROS")
    print(f"{'='*80}")
    
    # Carregar dataset original
    df = pd.read_excel(dataset_path)
    
    # Identificar classe
    class_column = 'Class' if 'Class' in df.columns else 'Classe'
    
    # Reconstruir DataFrame de teste (assumindo split com random_state=42)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), 
        test_size=0.2, 
        random_state=42, 
        stratify=df[class_column]
    )
    
    df_test = df.iloc[test_idx].copy()
    df_test['Predito'] = y_pred
    df_test['Real'] = y_test
    
    # Identificar erros
    df_test['Erro'] = df_test['Predito'] != df_test['Real']
    
    # Falsos Positivos (Real=0/Verdadeira, Predito=1/Falsa)
    falsos_positivos = df_test[(df_test['Real'] == 0) & (df_test['Predito'] == 1)]
    
    # Falsos Negativos (Real=1/Falsa, Predito=0/Verdadeira)
    falsos_negativos = df_test[(df_test['Real'] == 1) & (df_test['Predito'] == 0)]
    
    print(f"\nTotal de erros: {df_test['Erro'].sum()}")
    print(f"  Falsos Positivos: {len(falsos_positivos)}")
    print(f"  Falsos Negativos: {len(falsos_negativos)}")
    
    # Salvar exemplos
    campos_relevantes = ['Titulo', 'Subtitulo', 'Noticia', 'Real', 'Predito']
    campos_existentes = [c for c in campos_relevantes if c in df_test.columns]
    
    # Falsos Positivos
    if len(falsos_positivos) > 0:
        fp_sample = falsos_positivos[campos_existentes].head(n_exemplos)
        fp_file = os.path.join(output_dir, 'falsos_positivos_sfr.csv')
        fp_sample.to_csv(fp_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(fp_sample)} exemplos de Falsos Positivos salvos em: {fp_file}")
        
        # Exibir primeiro exemplo
        print(f"\n{'='*80}")
        print(f"EXEMPLO DE FALSO POSITIVO (Notícia VERDADEIRA classificada como FALSA)")
        print(f"{'='*80}")
        exemplo = falsos_positivos.iloc[0]
        print(f"Título: {exemplo.get('Titulo', 'N/A')}")
        print(f"Subtítulo: {exemplo.get('Subtitulo', 'N/A')}")
        if 'Noticia' in exemplo and pd.notna(exemplo['Noticia']):
            print(f"Corpo: {str(exemplo['Noticia'])[:300]}...")
        print(f"Real: {'Verdadeira' if exemplo['Real'] == 0 else 'Falsa'}")
        print(f"Predito: {'Verdadeira' if exemplo['Predito'] == 0 else 'Falsa'}")
    
    # Falsos Negativos
    if len(falsos_negativos) > 0:
        fn_sample = falsos_negativos[campos_existentes].head(n_exemplos)
        fn_file = os.path.join(output_dir, 'falsos_negativos_sfr.csv')
        fn_sample.to_csv(fn_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ {len(fn_sample)} exemplos de Falsos Negativos salvos em: {fn_file}")
        
        # Exibir primeiro exemplo
        print(f"\n{'='*80}")
        print(f"EXEMPLO DE FALSO NEGATIVO (Notícia FALSA classificada como VERDADEIRA)")
        print(f"{'='*80}")
        exemplo = falsos_negativos.iloc[0]
        print(f"Título: {exemplo.get('Titulo', 'N/A')}")
        print(f"Subtítulo: {exemplo.get('Subtitulo', 'N/A')}")
        if 'Noticia' in exemplo and pd.notna(exemplo['Noticia']):
            print(f"Corpo: {str(exemplo['Noticia'])[:300]}...")
        print(f"Real: {'Verdadeira' if exemplo['Real'] == 0 else 'Falsa'}")
        print(f"Predito: {'Verdadeira' if exemplo['Predito'] == 0 else 'Falsa'}")
    
    return falsos_positivos, falsos_negativos


def analisar_padroes_erros(falsos_positivos, falsos_negativos, output_dir):
    """Analisa padrões nos erros"""
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE PADRÕES NOS ERROS")
    print(f"{'='*80}")
    
    analise = {}
    
    # Analisar comprimento dos textos
    if len(falsos_positivos) > 0:
        fp_len_titulo = falsos_positivos['Titulo'].str.len().mean()
        fp_len_noticia = falsos_positivos['Noticia'].str.len().mean() if 'Noticia' in falsos_positivos else 0
        analise['FP_len_titulo'] = fp_len_titulo
        analise['FP_len_noticia'] = fp_len_noticia
        print(f"\nFalsos Positivos:")
        print(f"  Comprimento médio do título: {fp_len_titulo:.0f} caracteres")
        print(f"  Comprimento médio da notícia: {fp_len_noticia:.0f} caracteres")
    
    if len(falsos_negativos) > 0:
        fn_len_titulo = falsos_negativos['Titulo'].str.len().mean()
        fn_len_noticia = falsos_negativos['Noticia'].str.len().mean() if 'Noticia' in falsos_negativos else 0
        analise['FN_len_titulo'] = fn_len_titulo
        analise['FN_len_noticia'] = fn_len_noticia
        print(f"\nFalsos Negativos:")
        print(f"  Comprimento médio do título: {fn_len_titulo:.0f} caracteres")
        print(f"  Comprimento médio da notícia: {fn_len_noticia:.0f} caracteres")
    
    # Salvar análise
    analise_file = os.path.join(output_dir, 'analise_padroes_erros.txt')
    with open(analise_file, 'w', encoding='utf-8') as f:
        f.write("ANÁLISE DE PADRÕES NOS ERROS\n")
        f.write("="*80 + "\n\n")
        for key, value in analise.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✅ Análise de padrões salva em: {analise_file}")
    
    return analise


def gerar_relatorio_completo(best_result, cm_metrics, output_dir):
    """Gera relatório completo da análise"""
    
    relatorio_file = os.path.join(output_dir, 'relatorio_analise_erros.txt')
    
    with open(relatorio_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO COMPLETO DE ANÁLISE DE ERROS\n")
        f.write("SFR-Embedding-Mistral\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURAÇÃO DO MELHOR MODELO:\n")
        f.write("-"*80 + "\n")
        f.write(f"Dataset: {best_result['dataset']}\n")
        f.write(f"Combinação de campos: {best_result['combination']}\n")
        f.write(f"Classificador: {best_result['classifier']}\n")
        f.write(f"Dimensão dos embeddings: {best_result['embedding_dim']}\n\n")
        
        f.write("MÉTRICAS DE DESEMPENHO:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy: {best_result['accuracy']:.6f} ({best_result['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {best_result['precision']:.6f}\n")
        f.write(f"Recall: {best_result['recall']:.6f}\n")
        f.write(f"F1-Score: {best_result['f1_score']:.6f}\n\n")
        
        f.write("MATRIZ DE CONFUSÃO:\n")
        f.write("-"*80 + "\n")
        f.write(f"Verdadeiros Positivos (VP): {cm_metrics['TP']}\n")
        f.write(f"Verdadeiros Negativos (VN): {cm_metrics['TN']}\n")
        f.write(f"Falsos Positivos (FP): {cm_metrics['FP']} <- Notícias VERDADEIRAS classificadas como FALSAS\n")
        f.write(f"Falsos Negativos (FN): {cm_metrics['FN']} <- Notícias FALSAS classificadas como VERDADEIRAS\n\n")
        
        total_erros = cm_metrics['FP'] + cm_metrics['FN']
        total_amostras = cm_metrics['TP'] + cm_metrics['TN'] + cm_metrics['FP'] + cm_metrics['FN']
        taxa_erro = (total_erros / total_amostras) * 100
        
        f.write(f"RESUMO DE ERROS:\n")
        f.write(f"-"*80 + "\n")
        f.write(f"Total de erros: {total_erros} de {total_amostras} amostras\n")
        f.write(f"Taxa de erro: {taxa_erro:.2f}%\n")
        f.write(f"Taxa de acerto: {100-taxa_erro:.2f}%\n\n")
        
        f.write("ARQUIVOS GERADOS:\n")
        f.write("-"*80 + "\n")
        f.write("1. falsos_positivos_sfr.csv - Exemplos de FPs\n")
        f.write("2. falsos_negativos_sfr.csv - Exemplos de FNs\n")
        f.write("3. matriz_confusao_sfr.png - Visualização da matriz\n")
        f.write("4. analise_padroes_erros.txt - Análise de padrões\n")
        f.write("5. relatorio_analise_erros.txt - Este relatório\n")
    
    print(f"\n✅ Relatório completo salvo em: {relatorio_file}")


def main():
    """Função principal"""
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE ERROS - SFR-EMBEDDING-MISTRAL")
    print(f"{'='*80}\n")
    
    # 1. Carregar resultados
    all_results = carregar_resultados_completos()
    if all_results is None:
        return
    
    # 2. Encontrar melhor resultado
    best_result = encontrar_melhor_resultado(all_results)
    
    # 3. Gerar matriz de confusão
    y_test = best_result['y_test']
    y_pred = best_result['y_pred']
    
    cm, cm_metrics = gerar_matriz_confusao(y_test, y_pred, ANALYSIS_DIR)
    
    # 4. Extrair exemplos de erros
    dataset_path = f"../../bases_preprocessados/{best_result['dataset']}.xlsx"
    
    if os.path.exists(dataset_path):
        falsos_positivos, falsos_negativos = extrair_exemplos_erros(
            dataset_path, None, y_test, y_pred, ANALYSIS_DIR, n_exemplos=10
        )
        
        # 5. Analisar padrões
        analisar_padroes_erros(falsos_positivos, falsos_negativos, ANALYSIS_DIR)
    else:
        print(f"\n⚠️  Dataset original não encontrado: {dataset_path}")
        print(f"Ajuste o caminho na variável dataset_path")
    
    # 6. Gerar relatório completo
    gerar_relatorio_completo(best_result, cm_metrics, ANALYSIS_DIR)
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE CONCLUÍDA!")
    print(f"Todos os arquivos salvos em: {ANALYSIS_DIR}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
