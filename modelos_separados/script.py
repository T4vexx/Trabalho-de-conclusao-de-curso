
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# Carregar os CSVs
df_tfidf = pd.read_csv('melhores_hiperparametros_tfidf.csv')
df_sfr = pd.read_csv('melhores_hiperparametros_sfr.csv')
df_openai = pd.read_csv('melhores_hiperparametros_openai_gpu.csv')

# Adicionar coluna de Modelo de Embedding
df_tfidf['Embedding_Model'] = 'TF-IDF'
df_sfr['Embedding_Model'] = 'SFR-Mistral'
df_openai['Embedding_Model'] = 'OpenAI-3-small'

# Padronizar nomes de datasets
def padronizar_dataset(nome):
    if 'stopwords' in nome.lower():
        return 'Sem Stopwords'
    elif 'caracteres' in nome.lower() or 'especiais' in nome.lower():
        return 'Sem Caracteres Especiais'
    else:
        return nome

df_tfidf['Dataset_Clean'] = df_tfidf['Dataset'].apply(padronizar_dataset)
df_sfr['Dataset_Clean'] = df_sfr['Dataset'].apply(padronizar_dataset)
df_openai['Dataset_Clean'] = df_openai['Dataset'].apply(padronizar_dataset)

# Criar tabela resumida com MELHORES resultados de cada modelo
resultados_otimizados = []

for embedding_model, df in [('TF-IDF', df_tfidf), ('SFR-Mistral', df_sfr), ('OpenAI-3-small', df_openai)]:
    for dataset in df['Dataset_Clean'].unique():
        df_dataset = df[df['Dataset_Clean'] == dataset]
        
        # Melhor por classificador
        for classifier in df_dataset['Classifier'].unique():
            df_clf = df_dataset[df_dataset['Classifier'] == classifier]
            
            if len(df_clf) > 0:
                # Pegar o melhor F1-Score
                best_idx = df_clf['Test_F1'].idxmax()
                best_row = df_clf.loc[best_idx]
                
                resultado = {
                    'Embedding_Model': embedding_model,
                    'Dataset': dataset,
                    'Classifier': classifier,
                    'Method': best_row['Method'],
                    'Test_F1': best_row['Test_F1'],
                    'Test_Accuracy': best_row['Test_Accuracy'],
                    'Best_Params': best_row['Best_Params']
                }
                
                # Adicionar CV_Score se disponível (apenas TF-IDF tem)
                if 'CV_Score' in best_row:
                    resultado['CV_Score'] = best_row['CV_Score']
                else:
                    resultado['CV_Score'] = np.nan
                
                # GPU usado (SFR e OpenAI)
                if 'Used_GPU' in best_row:
                    resultado['Used_GPU'] = best_row['Used_GPU']
                else:
                    resultado['Used_GPU'] = False
                
                resultados_otimizados.append(resultado)

# Criar DataFrame
df_otimizados = pd.DataFrame(resultados_otimizados)

# Ordenar por F1-Score (decrescente)
df_otimizados = df_otimizados.sort_values('Test_F1', ascending=False).reset_index(drop=True)

# Formatar colunas
df_otimizados['Test_F1'] = df_otimizados['Test_F1'].round(4)
df_otimizados['Test_Accuracy'] = df_otimizados['Test_Accuracy'].round(4)
df_otimizados['CV_Score'] = df_otimizados['CV_Score'].round(4)

# Reordenar colunas
colunas_ordem = ['Embedding_Model', 'Dataset', 'Classifier', 'Method', 
                 'Test_F1', 'Test_Accuracy', 'CV_Score', 'Used_GPU', 'Best_Params']
df_otimizados = df_otimizados[colunas_ordem]

print("=" * 80)
print("RESUMO DOS MODELOS OTIMIZADOS (Top 10 por F1-Score):")
print("=" * 80)
print(df_otimizados[['Embedding_Model', 'Dataset', 'Classifier', 'Test_F1', 'Test_Accuracy']].head(10))

print("\n" + "=" * 80)
print("ESTATÍSTICAS POR MODELO DE EMBEDDING:")
print("=" * 80)
resumo_por_modelo = df_otimizados.groupby('Embedding_Model').agg({
    'Test_F1': ['mean', 'max', 'min'],
    'Test_Accuracy': ['mean', 'max', 'min']
}).round(4)
print(resumo_por_modelo)

# Salvar para usar na criação do Excel
df_otimizados.to_csv('modelos_otimizados_resumo.csv', index=False)
print("\n✅ Arquivo temporário salvo: modelos_otimizados_resumo.csv")
print(f"\n📊 Total de resultados: {len(df_otimizados)}")
