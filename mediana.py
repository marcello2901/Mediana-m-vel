import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.set_page_config(page_title="Medida Móvel - Lab Quality", layout="wide")

st.title("📊 Monitoramento de Medida Móvel Analítica")

# --- SIDEBAR: CONFIGURAÇÕES ---
st.sidebar.header("⚙️ Parâmetros Analíticos")
cva = st.sidebar.number_input("Pior Cenário CVa (%)", value=3.0) / 100
n_janela = st.sidebar.number_input("N (Tamanho da Janela)", value=10, min_value=2)
prob_confianca = 0.99999

# --- FUNÇÕES TÉCNICAS ---
def tratar_outliers_tukey(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    return serie.mask((serie < limite_inf) | (serie > limite_sup), serie.median())

def calcular_limite_estatistico(cva, n, prob):
    try:
        # INV.T(prob; n-1)
        fator_t = stats.t.ppf(prob, n-1)
        return fator_t * (0.75 * cva)
    except:
        return None

# --- CARREGAMENTO DE DADOS ---
uploaded_file = st.file_uploader("Suba sua planilha (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Tenta ler CSV com separadores comuns ou Excel
    try:
        if uploaded_file.name.endswith('.csv'):
            # O separador sep=None faz o pandas detectar se é vírgula ou ponto e vírgula sozinho
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("Arquivo carregado com sucesso!")
        
        # --- MAPEAMENTO DE COLUNAS PELO USUÁRIO ---
        st.subheader("🔗 Mapeamento de Colunas")
        col_list = df.columns.tolist()
        
        col_input_1, col_input_2, col_input_3 = st.columns(3)
        with col_input_1:
            col_data = st.selectbox("Coluna de Data", col_list)
        with col_input_2:
            col_hora = st.selectbox("Coluna de Hora", col_list)
        with col_input_3:
            col_res = st.selectbox("Coluna de Resultado (Numérico)", col_list)

        if st.button("Processar Dados"):
            # 1. Conversão e Ordenação Cronológica
            # Criamos uma coluna temporária combinando data e hora
            df['timestamp'] = pd.to_datetime(df[col_data].astype(str) + ' ' + df[col_hora].astype(str))
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            
            # Garantir que o resultado seja numérico
            df[col_res] = pd.to_numeric(df[col_res], errors='coerce')
            df = df.dropna(subset=[col_res]) # Remove linhas onde o resultado não é número

            # 2. Tratamento de Outliers (Tukey)
            df['res_limpo'] = tratar_outliers_tukey(df[col_res])

            # 3. Cálculos Móveis
            df['Mediana_Movel'] = df['res_limpo'].rolling(window=n_janela).median()
            df['DP_Movel'] = df['res_limpo'].rolling(window=n_janela).std()
            
            limite_sup_dp = calcular_limite_estatistico(cva, n_janela, prob_confianca)

            # --- VISUALIZAÇÃO ---
            st.divider()
            st.subheader(f"Análise de Mediana Móvel (N={n_janela})")
            
            st.line_chart(df.set_index('timestamp')['Mediana_Movel'])
            
            st.subheader("Controle de Dispersão (DP Móvel)")
            chart_data = df.set_index('timestamp')[['DP_Movel']]
            if limite_sup_dp:
                chart_data['Limite_Estatistico'] = limite_sup_dp
            st.line_chart(chart_data)

            if limite_sup_dp and (df['DP_Movel'] > limite_sup_dp).any():
                st.error("⚠️ Alerta: Desvio Padrão Móvel acima do limite estatístico detectado!")
                st.dataframe(df[df['DP_Movel'] > limite_sup_dp][[col_data, col_hora, col_res, 'DP_Movel']])

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
