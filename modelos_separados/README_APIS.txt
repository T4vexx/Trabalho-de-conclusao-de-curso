
╔══════════════════════════════════════════════════════════════════════════════╗
║           GUIA DE USO: OPENAI E GEMINI EMBEDDINGS                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 ARQUIVOS CRIADOS
════════════════════════════════════════════════════════════════════════════════

treinar_openai.py  → Usa OpenAI text-embedding-3-small
treinar_gemini.py  → Usa Google Gemini gemini-embedding-001

════════════════════════════════════════════════════════════════════════════════

🔑 CONFIGURAÇÃO DAS API KEYS
════════════════════════════════════════════════════════════════════════════════

OPENAI:
──────
1. Obtenha sua API key em: https://platform.openai.com/api-keys
2. Configure a variável de ambiente:

   Linux/Mac:
   export OPENAI_API_KEY='sk-proj-...'

   Windows (PowerShell):
   $env:OPENAI_API_KEY='sk-proj-...'

   Windows (CMD):
   set OPENAI_API_KEY=sk-proj-...

GEMINI:
──────
1. Obtenha sua API key em: https://aistudio.google.com/app/apikey
2. Configure a variável de ambiente:

   Linux/Mac:
   export GEMINI_API_KEY='AIza...'
   # ou
   export GOOGLE_API_KEY='AIza...'

   Windows (PowerShell):
   $env:GEMINI_API_KEY='AIza...'

   Windows (CMD):
   set GEMINI_API_KEY=AIza...

════════════════════════════════════════════════════════════════════════════════

📦 INSTALAÇÃO DE DEPENDÊNCIAS
════════════════════════════════════════════════════════════════════════════════

# Dependências comuns
pip install pandas numpy scikit-learn openpyxl tqdm

# Para OpenAI
pip install openai

# Para Gemini
pip install google-genai

# Instalar tudo de uma vez:
pip install pandas numpy scikit-learn openpyxl tqdm openai google-genai

════════════════════════════════════════════════════════════════════════════════

🚀 COMO EXECUTAR
════════════════════════════════════════════════════════════════════════════════

# OpenAI (text-embedding-3-small)
python treinar_openai.py

# Gemini (gemini-embedding-001)
python treinar_gemini.py

════════════════════════════════════════════════════════════════════════════════

💰 COMPARAÇÃO DE CUSTOS E CARACTERÍSTICAS
════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ OPENAI text-embedding-3-small                                                │
└─────────────────────────────────────────────────────────────────────────────┘
Custo: $0.02 por 1M tokens (~62,500 páginas por $1)
Dimensão padrão: 1536 (configurável até 512 sem perder muita qualidade)
Limite de tokens: 8,192 tokens por texto
Batch size: Até 2048 textos por request
Performance: 62.3% no MTEB benchmark
Vantagens:
  • Excelente custo-benefício
  • Rápido e eficiente
  • Suporta batch processing nativo
  • Boa performance multilíngue

┌─────────────────────────────────────────────────────────────────────────────┐
│ GEMINI gemini-embedding-001                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
Custo: Grátis até certo limite, depois $0.00001 por 1K caracteres
Dimensão padrão: 768 (recomendado: 768, 1536, 3072)
Limite de tokens: 2,048 tokens por texto
Batch size: Múltiplos textos por request
Task types: CLASSIFICATION, CLUSTERING, SEMANTIC_SIMILARITY, etc.
Vantagens:
  • API gratuita mais generosa
  • Task types específicos otimizam embeddings
  • Técnica Matryoshka (MRL) para dimensões flexíveis
  • Integração com ecossistema Google

════════════════════════════════════════════════════════════════════════════════

⚙️ OTIMIZAÇÕES IMPLEMENTADAS PARA ECONOMIA
════════════════════════════════════════════════════════════════════════════════

✓ Batch Processing:
  • OpenAI: 100 textos por request (limite: 2048)
  • Gemini: 100 textos por request
  • Reduz drasticamente o número de chamadas à API

✓ Truncamento de Texto:
  • Textos longos são truncados antes do envio
  • OpenAI: ~8000 chars (evita textos muito longos)
  • Gemini: ~2000 chars (limite mais restrito)

✓ Cache de Embeddings:
  • Embeddings salvos em arquivos .npy
  • Reutilizados se o script for executado novamente
  • Economiza custos ao re-treinar classificadores

✓ Dimensões Reduzidas:
  • OpenAI: 1536 dimensões (vs 1536 padrão)
  • Gemini: 768 dimensões (vs 3072 máximo)
  • Menos storage e mais rápido, sem perda significativa de qualidade

✓ Normalização (Gemini):
  • Embeddings normalizados para dimensões < 3072
  • Garante qualidade em dimensões reduzidas

✓ Rate Limiting:
  • Sleep entre requests para evitar rate limits
  • Fallback para processamento individual se batch falhar

════════════════════════════════════════════════════════════════════════════════

📊 ESTRUTURA DE SAÍDA
════════════════════════════════════════════════════════════════════════════════

results_openai/
  ├── resultados_openai.csv
  ├── resultados_completos_openai.pkl
  └── log_openai_YYYYMMDD_HHMMSS.txt

results_gemini/
  ├── resultados_gemini.csv
  ├── resultados_completos_gemini.pkl
  └── log_gemini_YYYYMMDD_HHMMSS.txt

embeddings_openai/
  └── embeddings_*.npy  (cache)

embeddings_gemini/
  └── embeddings_*.npy  (cache)

checkpoints_openai/
  └── checkpoint_openai.pkl

checkpoints_gemini/
  └── checkpoint_gemini.pkl

════════════════════════════════════════════════════════════════════════════════

🔄 SISTEMA DE CHECKPOINTS
════════════════════════════════════════════════════════════════════════════════

• Embeddings são salvos em cache (.npy)
• Se interromper, basta executar novamente
• Não gera embeddings duplicados (economiza $$$)
• Para reiniciar do zero:
  rm -rf embeddings_openai/ checkpoints_openai/
  rm -rf embeddings_gemini/ checkpoints_gemini/

════════════════════════════════════════════════════════════════════════════════

💡 DICAS PARA ECONOMIA
════════════════════════════════════════════════════════════════════════════════

1. USE O CACHE:
   • Nunca delete os embeddings_*/ se planeja re-treinar
   • Um embedding gerado = custo pago
   • Cache permite experimentar com diferentes classificadores sem custo extra

2. TESTE COM DATASET MENOR PRIMEIRO:
   • Pegue 100 linhas para testar
   • Verifique se tudo funciona
   • Depois processe o dataset completo

3. AJUSTE AS DIMENSÕES:
   • OpenAI: 512 ou 1536 são suficientes para a maioria dos casos
   • Gemini: 768 tem excelente custo-benefício

4. TRUNCAMENTO INTELIGENTE:
   • Textos processados/lemmatizados são mais curtos
   • Menos tokens = menos custo
   • Por isso usamos os datasets processados!

5. GEMINI PARA TESTES, OPENAI PARA PRODUÇÃO:
   • Gemini tem API gratuita mais generosa
   • Use para experimentação
   • OpenAI para casos que exigem escala

════════════════════════════════════════════════════════════════════════════════

⏱️ TEMPO ESTIMADO DE EXECUÇÃO
════════════════════════════════════════════════════════════════════════════════

Para ~10,000 textos:

OpenAI:
  • Com cache: ~5 minutos (só classificadores)
  • Sem cache: ~15-20 minutos (geração de embeddings + classificadores)

Gemini:
  • Com cache: ~5 minutos
  • Sem cache: ~20-30 minutos (rate limits mais conservadores)

Obs: Tempo varia com velocidade da conexão e rate limits das APIs

════════════════════════════════════════════════════════════════════════════════

🐛 TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════════

Erro: "API key not found"
  → Verifique se configurou a variável de ambiente
  → Teste: echo $OPENAI_API_KEY ou echo $GEMINI_API_KEY

Erro: "Rate limit exceeded"
  → Normal, o script tem retry automático
  → Aguarde alguns segundos e tente novamente
  → Reduza BATCH_SIZE no código se persistir

Erro: "Token limit exceeded"
  → Textos muito longos
  → Ajuste max_length na função truncate_text()

Embeddings muito lentos:
  → Verifique sua conexão com internet
  → APIs podem estar com latência alta
  → Use cache para evitar regenerar

Custo muito alto:
  → SEMPRE use os embeddings em cache
  → Não delete embeddings_*/ sem necessidade
  → Considere usar apenas parte do dataset para testes

════════════════════════════════════════════════════════════════════════════════

📈 EXEMPLO DE USO DOS RESULTADOS
════════════════════════════════════════════════════════════════════════════════

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar todos os resultados
df_openai = pd.read_csv('results_openai/resultados_openai.csv')
df_gemini = pd.read_csv('results_gemini/resultados_gemini.csv')

# Adicionar coluna de método
df_openai['Embedding'] = 'OpenAI'
df_gemini['Embedding'] = 'Gemini'

# Combinar
df_all = pd.concat([df_openai, df_gemini])

# Plotar comparação
plt.figure(figsize=(14, 6))
sns.barplot(data=df_all, x='Combination', y='F1-Score', hue='Embedding')
plt.xticks(rotation=45)
plt.title('Comparação OpenAI vs Gemini - F1-Score por Combinação')
plt.tight_layout()
plt.savefig('comparacao_apis.png', dpi=300)

════════════════════════════════════════════════════════════════════════════════

📞 COMANDOS RÁPIDOS
════════════════════════════════════════════════════════════════════════════════

# Configurar API keys
export OPENAI_API_KEY='sua-chave'
export GEMINI_API_KEY='sua-chave'

# Instalar dependências
pip install pandas numpy scikit-learn openpyxl tqdm openai google-genai

# Executar
python treinar_openai.py
python treinar_gemini.py

# Ver logs
cat results_openai/log_openai_*.txt
cat results_gemini/log_gemini_*.txt

# Limpar cache (CUIDADO: vai regenerar embeddings = custo!)
rm -rf embeddings_openai/ checkpoints_openai/
rm -rf embeddings_gemini/ checkpoints_gemini/

════════════════════════════════════════════════════════════════════════════════
