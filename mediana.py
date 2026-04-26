import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.set_page_config(page_title="Medida Móvel - Lab Quality", layout="wide")

st.title("📊 Monitoramento de Medida Móvel Analítica")

# --- SIDEBAR: ESPECIFICAÇÕES DA QUALIDADE ---
st.sidebar.header("⚙️ Configurações de Base")
cvi = st.sidebar.number_input("CVi (%)", value=2.0, step=0.1) / 100
cvg = st.sidebar.number_input("CVg (%)", value=5.0, step=0.1) / 100
cva = st.sidebar.number_input("Pior Cenário CVa (%)", value=3.0, step=0.1) / 100

st.sidebar.divider()
st.sidebar.subheader("Seleção de Limites")

# Cálculo Prévio das Opções (Baseado na planilha)
# Nota: As fórmulas abaixo seguem o padrão de metas de Fraser/Milan
op1 = (0.5 * cvi) * 100  # Desejável
op2 = (0.25 * cvi) * 100 # Mínima
op5 = 0.25 * np.sqrt(cvi**2 + cvg**2) * 100 # Erro Sist. Desejável
op6 = 0.375 * np.sqrt(cvi**2 + cvg**2) * 100 # Erro Sist. Mínima
rcv = 2.77 * np.sqrt(cva**2 + cvi**2) # Opção 4 (RCV)

# Interface de escolha das Opções
escolha_limite = st.sidebar.selectbox(
    "Escolha a meta para Mediana Móvel:",
    [
        f"Opção 1 - EA Desejável ({op1:.2f}%)",
        f"Opção 2 - EA Mínima ({op2:.2f}%)",
        "Opção 3 - Outra Fonte (EA)",
        f"Opção 4 - RCV ({rcv:.2f} absoluto)",
        f"Opção 5 - ES Desejável ({op5:.2f}%)",
        f"Opção 6 - ES Mínima ({op6:.2f}%)",
        "Opção 7 - Outra Fonte (ES)"
    ]
)

# Input para valores manuais (Opção 3 e 7)
valor_custom = 0.0
if "Opção 3" in escolha_limite or "Opção 7" in escolha_limite:
    valor_custom = st.sidebar.number_input("Defina o valor customizado (%)", value=10.0) / 100

st.sidebar.divider()
n_janela = st.sidebar.number_input("N (Tamanho da Janela)", value=10, min_value=2)
prob_confianca = 0.99999

# --- FUNÇÕES TÉCNICAS ---
def tratar_outliers_tukey(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    # Substitui pelo valor da mediana da série
    return serie.mask((serie < limite_inf) | (serie > limite_sup), serie.median())

def obter_valor_limite(escolha, custom, op1, op2, rcv, op5, op6):
    if "Opção 1" in escolha: return op1 / 100
    if "Opção 2" in escolha: return op2 / 100
    if "Opção 4" in escolha: return rcv
    if "Opção 5" in escolha: return op5 / 100
    if "Opção 6" in escolha: return op6 / 100
    return custom

# --- CARREGAMENTO E PROCESSAMENTO ---
uploaded_file = st.file_uploader("Suba sua planilha (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("Arquivo carregado!")
        
        col_list = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: col_data = st.selectbox("Data", col_list)
        with c2: col_hora = st.selectbox("Hora", col_list)
        with c3: col_res = st.selectbox("Resultado", col_list)

        if st.button("Executar Análise"):
            # Limpeza de Dados (Vírgula para Ponto)
            df[col_res] = df[col_res].astype(str).str.replace(',', '.').str.strip()
            df[col_res] = pd.to_numeric(df[col_res], errors='coerce')
            df = df.dropna(subset=[col_res])

            # Ordenação Cronológica
            df['timestamp'] = pd.to_datetime(df[col_data].astype(str) + ' ' + df[col_hora].astype(str), dayfirst=True)
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            # Tratamento de Outliers
            df['res_limpo'] = tratar_outliers_tukey(df[col_res])

            # Cálculos Móveis (O segredo do gráfico oscilante)
            df['Mediana_Movel'] = df['res_limpo'].rolling(window=n_janela, min_periods=1).median()
            df['DP_Movel'] = df['res_limpo'].rolling(window=n_janela, min_periods=2).std()

            # Definição do Limite da Mediana (Base Alvo)
            alvo_estatistico = df['res_limpo'].median()
            limite_percentual = obter_valor_limite(escolha_limite, valor_custom, op1, op2, rcv, op5, op6)
            
            # Se for RCV (Opção 4), o limite é absoluto, senão é percentual sobre o alvo
            if "Opção 4" in escolha_limite:
                df['LSC_Mediana'] = alvo_estatistico + limite_percentual
                df['LIC_Mediana'] = alvo_estatistico - limite_percentual
            else:
                df['LSC_Mediana'] = alvo_estatistico * (1 + limite_percentual)
                df['LIC_Mediana'] = alvo_estatistico * (1 - limite_percentual)

            # Limite do DP (Fator t-Student 99.999%)
            fator_t = stats.t.ppf(prob_confianca, n_janela-1)
            limite_dp = fator_t * (0.75 * cva)

            # --- GRÁFICOS ---
            st.divider()
            st.subheader("📈 Gráfico de Mediana Móvel")
            # Plotando Mediana + Limites de Controle
            st.line_chart(df.set_index('timestamp')[['Mediana_Movel', 'LSC_Mediana', 'LIC_Mediana']])

            st.subheader("📉 Gráfico de Precisão (DP Móvel)")
            df['Limite_DP'] = limite_dp
            st.line_chart(df.set_index('timestamp')[['DP_Movel', 'Limite_DP']])

            # Tabela de Alertas
            alertas = df[(df['Mediana_Movel'] > df['LSC_Mediana']) | 
                        (df['Mediana_Movel'] < df['LIC_Mediana']) |
                        (df['DP_Movel'] > limite_dp)]
            
            if not alertas.empty:
                st.warning(f"Foram detectados {len(alertas)} pontos fora dos limites!")
                st.dataframe(alertas[[col_data, col_hora, col_res, 'Mediana_Movel', 'DP_Movel']])

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
