import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Medida Móvel - Lab Quality", layout="wide")

st.title("📊 Monitoramento de Medida Móvel Analítica")
st.markdown("""
Esta aplicação transpõe foca em Mediana Móvel e DP Móvel Robusto.
""")

# --- SIDEBAR: ESPECIFICAÇÕES DA QUALIDADE ANALÍTICA ---
st.sidebar.header("⚙️ Especificações da Qualidade")
cvi = st.sidebar.number_input("CVi (%)", value=2.0) / 100
cvg = st.sidebar.number_input("CVg (%)", value=2.0) / 100
cva = st.sidebar.number_input("Pior Cenário CVa (%)", value=3.0) / 100

# Parâmetros para o Limite Superior do DP Móvel
st.sidebar.subheader("Limites Estatísticos")
prob_confianca = 0.99999
n_amostras = st.sidebar.number_input("N (Janela da Média)", value=10)

# --- FUNÇÕES CORE (LÓGICA DA PLANILHA) ---

def tratar_outliers_tukey(data):
    """Implementa a detecção e substituição de outliers pelo teste Tukey."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    
    # Substituição pela mediana (comum em algoritmos robustos)
    mediana = data.median()
    data_limpa = data.mask((data < limite_inf) | (data > limite_sup), mediana)
    return data_limpa

def calcular_limite_dp(cva, n, prob):
    """
    Transpõe a fórmula: =SEERRO(INV.T(prob; n-1)*(0,75*CVa); "")
    """
    try:
        fator_t = stats.t.ppf(prob, n-1)
        limite = fator_t * (0.75 * cva)
        return limite
    except:
        return None

# --- CARREGAMENTO DE DADOS ---
uploaded_file = st.file_uploader("Suba o CSV com os resultados dos pacientes", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Supondo coluna 'Resultado'
    if 'Resultado' in df.columns:
        
        # 1. Tratamento de Outliers
        df['Resultado_Tratado'] = tratar_outliers_tukey(df['Resultado'])
        
        # 2. Cálculos Móveis
        df['Mediana_Movel'] = df['Resultado_Tratado'].rolling(window=n_amostras).median()
        df['DP_Movel'] = df['Resultado_Tratado'].rolling(window=n_amostras).std()
        
        # 3. Limites (Baseados na sua fórmula INV.T)
        limite_sup_dp = calcular_limite_dp(cva, n_amostras, prob_confianca)
        
        # --- EXIBIÇÃO ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tendência da Mediana")
            st.line_chart(df['Mediana_Movel'])
            
        with col2:
            st.subheader("Controle de Precisão (DP Móvel)")
            st.line_chart(df['DP_Movel'])
            if limite_sup_dp:
                st.warning(f"Limite Superior Estatístico do DP: {limite_sup_dp:.4f}")

        # Identificação de Alertas
        df['Alerta_DP'] = df['DP_Movel'] > limite_sup_dp
        if df['Alerta_DP'].any():
            st.error("⚠️ Alerta: O Desvio Padrão Móvel ultrapassou o limite estatístico em alguns pontos!")
            st.dataframe(df[df['Alerta_DP'] == True])
            
    else:
        st.error("O arquivo deve conter uma coluna chamada 'Resultado'.")

else:
    st.info("Aguardando upload de dados para iniciar a análise robusta.")
