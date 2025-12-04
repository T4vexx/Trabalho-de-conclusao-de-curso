import pandas as pd
import os
from pathlib import Path
import numpy as np

def encontrar_resultados_csv(diretorio_base):
    """
    Procura recursivamente por arquivos resultados_*.csv
    """
    arquivos_encontrados = []

    # Padrões de busca mais flexíveis
    padroes = [
        '*/results_*/resultados_*.csv',  # Padrão normal
        '*/resultados_*.csv',             # Direto na pasta do modelo
        'results_*/resultados_*.csv'      # Na raiz
    ]

    for padrao in padroes:
        for arquivo_csv in Path(diretorio_base).glob(padrao):
            # Tentar extrair nome do modelo
            partes = arquivo_csv.parts

            # Procurar por pasta de modelo (que não seja results_*)
            nome_modelo = None
            for parte in reversed(partes):
                if not parte.startswith('results_') and parte != '.':
                    nome_modelo = parte
                    break

            # Se não encontrou, usar nome do arquivo
            if not nome_modelo:
                nome_modelo = arquivo_csv.stem.replace('resultados_', '')

            arquivos_encontrados.append((nome_modelo, arquivo_csv))

    # Remover duplicatas
    arquivos_unicos = list(set(arquivos_encontrados))
    arquivos_unicos.sort(key=lambda x: x[0])

    return arquivos_unicos

def analisar_por_classificador(df):
    """Análise comparativa por classificador"""
    print("\n" + "="*80)
    print("ANÁLISE POR CLASSIFICADOR:")
    print("="*80)

    analises = []
    for clf in df['Classifier'].unique():
        df_clf = df[df['Classifier'] == clf]
        melhor = df_clf.loc[df_clf['F1-Score'].idxmax()]

        analises.append({
            'Classificador': clf,
            'F1-Score Médio': df_clf['F1-Score'].mean(),
            'F1-Score Máximo': df_clf['F1-Score'].max(),
            'Melhor Modelo': melhor['Model'],
            'Accuracy Média': df_clf['Accuracy'].mean()
        })

        print(f"\n📊 {clf}:")
        print(f"   F1-Score médio: {df_clf['F1-Score'].mean():.4f}")
        print(f"   F1-Score melhor: {df_clf['F1-Score'].max():.4f} ({melhor['Model']})")
        print(f"   Accuracy média: {df_clf['Accuracy'].mean():.4f}")

    return pd.DataFrame(analises)

def analisar_por_combinacao(df):
    """Análise comparativa por combinação de campos"""
    print("\n" + "="*80)
    print("ANÁLISE POR COMBINAÇÃO DE CAMPOS:")
    print("="*80)

    analises = []
    for combo in ['titulo', 'texto', 'subtitulo', 'titulo_subtitulo', 'completo']:
        if combo in df['Combination'].values:
            df_combo = df[df['Combination'] == combo]
            melhor = df_combo.loc[df_combo['F1-Score'].idxmax()]

            analises.append({
                'Combinação': combo,
                'F1-Score Médio': df_combo['F1-Score'].mean(),
                'F1-Score Máximo': df_combo['F1-Score'].max(),
                'Melhor Modelo': melhor['Model'],
                'Melhor Classificador': melhor['Classifier']
            })

            print(f"\n📊 {combo}:")
            print(f"   F1-Score médio: {df_combo['F1-Score'].mean():.4f}")
            print(f"   Melhor: {melhor['F1-Score']:.4f} ({melhor['Model']} + {melhor['Classifier']})")

    return pd.DataFrame(analises)

def analisar_por_dataset(df):
    """Análise comparativa por dataset"""
    print("\n" + "="*80)
    print("ANÁLISE POR DATASET:")
    print("="*80)

    analises = []
    for dataset in df['Dataset'].unique():
        df_ds = df[df['Dataset'] == dataset]
        melhor = df_ds.loc[df_ds['F1-Score'].idxmax()]

        analises.append({
            'Dataset': dataset,
            'F1-Score Médio': df_ds['F1-Score'].mean(),
            'F1-Score Máximo': df_ds['F1-Score'].max(),
            'Melhor Modelo': melhor['Model'],
            'Melhor Combinação': melhor['Combination']
        })

        print(f"\n📊 {dataset}:")
        print(f"   F1-Score médio: {df_ds['F1-Score'].mean():.4f}")
        print(f"   Melhor: {melhor['F1-Score']:.4f} ({melhor['Model']} + {melhor['Combination']} + {melhor['Classifier']})")

    return pd.DataFrame(analises)

def criar_tabela_comparativa_modelos(df):
    """Cria tabela pivot comparando todos os modelos"""
    print("\n" + "="*80)
    print("TABELA COMPARATIVA - MODELOS (Melhor F1 por combinação):")
    print("="*80)

    # Para cada combinação, pegar o melhor resultado de cada modelo
    resultados = []

    for modelo in sorted(df['Model'].unique()):
        df_modelo = df[df['Model'] == modelo]
        linha = {'Model': modelo}

        for combo in ['titulo', 'texto', 'subtitulo', 'titulo_subtitulo', 'completo']:
            df_combo = df_modelo[df_modelo['Combination'] == combo]
            if len(df_combo) > 0:
                linha[combo] = df_combo['F1-Score'].max()
            else:
                linha[combo] = np.nan

        # Média geral
        linha['MÉDIA'] = df_modelo['F1-Score'].mean()
        resultados.append(linha)

    df_comparacao = pd.DataFrame(resultados)
    df_comparacao = df_comparacao.sort_values('MÉDIA', ascending=False)

    print("\n" + df_comparacao.to_string(index=False, float_format='%.4f'))

    return df_comparacao

def consolidar_resultados(diretorio_base, arquivo_saida='resultados_consolidados.xlsx'):
    """
    Consolida todos os resultados em um único XLSX com múltiplas abas
    """
    print("="*80)
    print("CONSOLIDADOR AVANÇADO DE RESULTADOS - EXCEL (XLSX)")
    print("="*80)

    # Encontrar todos os arquivos
    print(f"\nBuscando arquivos em: {os.path.abspath(diretorio_base)}")
    arquivos = encontrar_resultados_csv(diretorio_base)

    if not arquivos:
        print("\n❌ Nenhum arquivo de resultados encontrado!")
        print("\n📁 Estrutura esperada:")
        print("  MODELOS_SEPARADOS/")
        print("    ├── alibaba/")
        print("    │   └── results_gte/")
        print("    │       └── resultados_gte.csv")
        print("    ├── bert/")
        print("    │   └── results_bert/")
        print("    │       └── resultados_bert.csv")
        print("    └── ...")
        return None

    print(f"\n✅ Encontrados {len(arquivos)} arquivo(s):")
    for modelo, caminho in arquivos:
        print(f"  • {modelo:25s} → {caminho.name}")

    # Lista para armazenar todos os dataframes
    dfs = []
    modelos_processados = []
    modelos_com_erro = []

    print("\n" + "="*80)
    print("PROCESSANDO ARQUIVOS:")
    print("="*80)

    # Processar cada arquivo
    for nome_modelo, arquivo_csv in arquivos:
        try:
            print(f"\n📥 {nome_modelo}...")
            df = pd.read_csv(arquivo_csv)

            # Verificar colunas necessárias
            colunas_esperadas = ['Dataset', 'Combination', 'Classifier', 'Accuracy', 
                               'Precision', 'Recall', 'F1-Score', 'Embedding_Dim']

            colunas_faltando = [col for col in colunas_esperadas if col not in df.columns]
            if colunas_faltando:
                print(f"  ⚠️  Colunas faltando: {colunas_faltando}")

            # Adicionar coluna com nome do modelo
            df.insert(0, 'Model', nome_modelo)

            # Informações do modelo
            print(f"  ✓ {len(df)} linhas")
            print(f"  ✓ Dimensão embedding: {df['Embedding_Dim'].iloc[0] if 'Embedding_Dim' in df.columns else 'N/A'}")
            print(f"  ✓ F1-Score: {df['F1-Score'].min():.4f} - {df['F1-Score'].max():.4f}")

            dfs.append(df)
            modelos_processados.append(nome_modelo)

        except Exception as e:
            print(f"  ❌ Erro: {e}")
            modelos_com_erro.append(nome_modelo)
            continue

    if not dfs:
        print("\n❌ Nenhum arquivo foi processado com sucesso!")
        return None

    # Concatenar todos os dataframes
    print("\n" + "="*80)
    print("CONSOLIDANDO DADOS:")
    print("="*80)

    df_consolidado = pd.concat(dfs, ignore_index=True)

    # Ordenar
    df_consolidado = df_consolidado.sort_values(
        by=['Model', 'Dataset', 'Combination', 'F1-Score'],
        ascending=[True, True, True, False]
    )

    # ESTATÍSTICAS GERAIS
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS GERAIS:")
    print("="*80)

    print(f"\n✓ Total de experimentos: {len(df_consolidado)}")
    print(f"✓ Modelos processados: {len(modelos_processados)}")
    print(f"✓ Datasets: {df_consolidado['Dataset'].nunique()}")
    print(f"✓ Combinações: {df_consolidado['Combination'].nunique()}")
    print(f"✓ Classificadores: {df_consolidado['Classifier'].nunique()}")

    if modelos_com_erro:
        print(f"\n⚠️  Modelos com erro: {', '.join(modelos_com_erro)}")

    print(f"\n📈 Métricas globais:")
    print(f"  • F1-Score:  {df_consolidado['F1-Score'].mean():.4f} (média) | "
          f"{df_consolidado['F1-Score'].max():.4f} (max)")
    print(f"  • Accuracy:  {df_consolidado['Accuracy'].mean():.4f} (média) | "
          f"{df_consolidado['Accuracy'].max():.4f} (max)")
    print(f"  • Precision: {df_consolidado['Precision'].mean():.4f} (média)")
    print(f"  • Recall:    {df_consolidado['Recall'].mean():.4f} (média)")

    # ANÁLISES DETALHADAS
    df_analise_clf = analisar_por_classificador(df_consolidado)
    df_analise_combo = analisar_por_combinacao(df_consolidado)
    df_analise_dataset = analisar_por_dataset(df_consolidado)

    # TOP RESULTADOS
    print("\n" + "="*80)
    print("🏆 TOP 15 MELHORES RESULTADOS (F1-Score):")
    print("="*80)

    top15 = df_consolidado.nlargest(15, 'F1-Score')
    print("\n" + top15[['Model', 'Dataset', 'Combination', 'Classifier', 
                        'F1-Score', 'Accuracy']].to_string(index=False))

    # MELHOR POR MODELO
    print("\n" + "="*80)
    print("🥇 MELHOR RESULTADO POR MODELO:")
    print("="*80)

    melhor_por_modelo = df_consolidado.loc[df_consolidado.groupby('Model')['F1-Score'].idxmax()]
    melhor_por_modelo = melhor_por_modelo.sort_values('F1-Score', ascending=False)

    print("\n" + melhor_por_modelo[['Model', 'F1-Score', 'Accuracy', 
                                     'Dataset', 'Combination', 'Classifier']].to_string(index=False))

    # TABELA COMPARATIVA
    df_comparacao = criar_tabela_comparativa_modelos(df_consolidado)

    # PIOR RESULTADO (para análise)
    print("\n" + "="*80)
    print("📉 PIORES 5 RESULTADOS (para análise):")
    print("="*80)

    piores = df_consolidado.nsmallest(5, 'F1-Score')
    print("\n" + piores[['Model', 'Dataset', 'Combination', 'Classifier', 
                         'F1-Score', 'Accuracy']].to_string(index=False))

    # SALVAR TUDO EM UM ÚNICO ARQUIVO EXCEL COM MÚLTIPLAS ABAS
    print("\n" + "="*80)
    print("💾 SALVANDO ARQUIVO EXCEL:")
    print("="*80)

    with pd.ExcelWriter(arquivo_saida, engine='openpyxl') as writer:
        # Aba 1: Dados Consolidados
        df_consolidado.to_excel(writer, sheet_name='Consolidado', index=False)
        print(f"  ✓ Aba 'Consolidado' - Todos os resultados")

        # Aba 2: Resumo por Modelo
        melhor_por_modelo.to_excel(writer, sheet_name='Resumo Modelos', index=False)
        print(f"  ✓ Aba 'Resumo Modelos' - Melhor de cada modelo")

        # Aba 3: Comparação Modelos
        df_comparacao.to_excel(writer, sheet_name='Comparação Modelos', index=False)
        print(f"  ✓ Aba 'Comparação Modelos' - Tabela comparativa")

        # Aba 4: Top 15
        top15.to_excel(writer, sheet_name='Top 15', index=False)
        print(f"  ✓ Aba 'Top 15' - Melhores resultados")

        # Aba 5: Análise por Classificador
        df_analise_clf.to_excel(writer, sheet_name='Análise Classificador', index=False)
        print(f"  ✓ Aba 'Análise Classificador'")

        # Aba 6: Análise por Combinação
        df_analise_combo.to_excel(writer, sheet_name='Análise Combinação', index=False)
        print(f"  ✓ Aba 'Análise Combinação'")

        # Aba 7: Análise por Dataset
        df_analise_dataset.to_excel(writer, sheet_name='Análise Dataset', index=False)
        print(f"  ✓ Aba 'Análise Dataset'")

    print(f"\n✅ Arquivo salvo: {arquivo_saida}")

    return df_consolidado

if __name__ == "__main__":
    # Configuração
    DIRETORIO_BASE = '.'  # Pasta MODELOS_SEPARADOS
    ARQUIVO_SAIDA = 'resultados_todos_modelos_consolidado.xlsx'

    print("\n🚀 Iniciando consolidação de resultados...")
    print(f"📁 Diretório: {os.path.abspath(DIRETORIO_BASE)}")
    print()

    # Executar
    df = consolidar_resultados(DIRETORIO_BASE, ARQUIVO_SAIDA)

    if df is not None:
        print("\n" + "="*80)
        print("✅ CONSOLIDAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*80)

        print(f"\n📄 Arquivo Excel gerado:")
        print(f"  {ARQUIVO_SAIDA}")

        print(f"\n📑 Abas incluídas:")
        print(f"  1. Consolidado - Todos os resultados")
        print(f"  2. Resumo Modelos - Melhor de cada modelo")
        print(f"  3. Comparação Modelos - Tabela comparativa")
        print(f"  4. Top 15 - Melhores resultados")
        print(f"  5. Análise Classificador - Por SVM/RF/LR")
        print(f"  6. Análise Combinação - Por tipo de campo")
        print(f"  7. Análise Dataset - Por dataset")

        print("\n💡 Próximos passos:")
        print("  • Abra o arquivo no Excel/LibreOffice")
        print("  • Navegue entre as abas para diferentes visões")
        print("  • Use filtros e tabelas dinâmicas")
        print("  • Crie gráficos comparativos")
        print("  • Perfeito para análise no seu TCC!")

    else:
        print("\n❌ Falha na consolidação!")
        print("\n🔍 Verifique:")
        print("  1. O script está na pasta MODELOS_SEPARADOS?")
        print("  2. As pastas dos modelos existem?")
        print("  3. Os arquivos resultados_*.csv estão dentro de results_*/?")
        print("  4. Você tem openpyxl instalado? (pip install openpyxl)")
