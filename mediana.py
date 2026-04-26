import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Configuração de Layout e Performance
st.set_page_config(page_title="Medida Móvel - ISO 13528 & Metas Clínicas", layout="wide")

# --- BLOCO 1: ALGORITMOS ROBUSTOS (ISO 13528) ---

@st.cache_data
def algoritmo_a_robusto(serie, max_iter=25):
    """
    Implementação do Algoritmo A (ISO 13528:2005) para Média e DP Robustos.
    Utilizado para definir os limites estatísticos de variação do equipamento.
    """
    x = serie.values
    mu = np.median(x)
    s = 1.483 * np.median(np.abs(x - mu))
    
    for _ in range(max_iter):
        delta = 1.5 * s
        x_cap = np.clip(x, mu - delta, mu + delta)
        mu_new = np.mean(x_cap)
        s_new = 1.134 * np.std(x_cap, ddof=1)
        
        if abs(mu - mu_new) < 1e-6 and abs(s - s_new) < 1e-6:
            break
        mu, s = mu_new, s_new
        
    return mu, s

def tratar_outliers_tukey(serie):
    """Detecção e substituição de outliers pelo teste de Tukey (1.5 * IQR)."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    return serie.mask((serie < lim_inf) | (serie > lim_sup), serie.median())

# --- BLOCO 2: PROCESSAMENTO DE DADOS ---

@st.cache_data
def processar_estatistica_completa(df_raw, col_res, col_data, col_hora, n_janela):
    df = df_raw.copy()
    
    # Limpeza de dados brasileiros (vírgula por ponto)
    df[col_res] = pd.to_numeric(df[col_res].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
    df = df.dropna(subset=[col_res])
    
    # Ordenação Cronológica Estrita
    df['timestamp'] = pd.to_datetime(df[col_data].astype(str) + ' ' + df[col_hora].astype(str), dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 1. Tratamento de Outliers para Mediana Móvel
    df['res_limpo'] = tratar_outliers_tukey(df[col_res])
    
    # 2. Cálculos Móveis
    df['Mediana_Movel'] = df['res_limpo'].rolling(window=int(n_janela), min_periods=1).median()
    df['DP_Movel'] = df['res_limpo'].rolling(window=int(n_janela), min_periods=2).std().fillna(0)
    
    # 3. Parâmetros Robustos (ISO 13528) para Limites Estatísticos
    mu_r, s_r = algoritmo_a_robusto(df[col_res])
    
    # 4. Limites Estatísticos (Estabilidade Técnica do Processo)
    # Reflete a variação esperada do instrumento (3 * Erro Padrão da Mediana)
    erro_padrao_mediana = (1.2533 * s_r) / np.sqrt(n_janela) # Ajuste para Mediana
    df['LSC_Est'] = mu_r + (3 * erro_padrao_mediana)
    df['LIC_Est'] = mu_r - (3 * erro_padrao_mediana)
    
    return df, mu_r, s_r

# --- BLOCO 3: INTERFACE E GRÁFICOS ---

st.title("📊 Ferramenta Medida Móvel - Qualidade Avançada")
st.markdown("Monitoramento baseado em **ISO 13528** (Limites Estatísticos) e **Variação Biológica** (Limites Clínicos).")

with st.sidebar:
    st.header("⚙️ Parâmetros Analíticos")
    cvi = st.number_input("CVi (%)", value=2.0) / 100
    cvg = st.number_input("CVg (%)", value=5.0) / 100
    cva = st.number_input("CVa (Pior Cenário) (%)", value=3.0) / 100
    
    st.divider()
    st.subheader("Configuração de Metas Clínicas")
    
    # Cálculos das Opções de Meta
    op1_ea = (0.5 * cvi) * 100
    op2_ea = (0.25 * cvi) * 100
    rcv = 2.77 * np.sqrt(cva**2 + cvi**2)
    op5_es = 0.25 * np.sqrt(cvi**2 + cvg**2) * 100
    op6_es = 0.375 * np.sqrt(cvi**2 + cvg**2) * 100

    meta_selecionada = st.selectbox("Selecione a Meta Clínica (Mediana):", [
        f"Opção 1 - EA Desejável ({op1_ea:.2f}%)",
        f"Opção 2 - EA Mínima ({op2_ea:.2f}%)",
        "Opção 3 - Outra Fonte (EA)",
        f"Opção 4 - RCV ({rcv:.2f} Absoluto)",
        f"Opção 5 - ES Desejável ({op5_es:.2f}%)",
        f"Opção 6 - ES Mínima ({op6_es:.2f}%)",
        "Opção 7 - Outra Fonte (ES)"
    ])
    
    custom_val = 0.0
    if "Opção 3" in meta_selecionada or "Opção 7" in meta_selecionada:
        custom_val = st.number_input("Valor Customizado (%)", value=5.0) / 100

    n_janela = st.number_input("N (Tamanho da Janela)", value=20, min_value=2)

# Upload de Arquivo
uploaded_file = st.file_uploader("Suba sua planilha de resultados (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Carregamento Inicial
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        df_input = pd.read_excel(uploaded_file)
    
    st.write(f"Linhas carregadas: {len(df_input)}")
    
    # Mapeamento de Colunas
    cols = df_input.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1: c_data = st.selectbox("Data do Resultado", cols)
    with c2: c_hora = st.selectbox("Hora do Resultado", cols)
    with c3: c_res = st.selectbox("Valor do Resultado", cols)

    if st.button("🚀 Executar Processamento Robusto"):
        with st.spinner("Aplicando Algoritmos ISO 13528 e Calculando Metas..."):
            
            # Processar Estatísticas
            df_final, mu_robusto, s_robusto = processar_estatistica_completa(df_input, c_res, c_data, c_hora, n_janela)
            
            # Cálculo dos Limites Clínicos (Aba de Metas)
            if "Opção 1" in meta_selecionada: m_val = op1_ea / 100
            elif "Opção 2" in meta_selecionada: m_val = op2_ea / 100
            elif "Opção 5" in meta_selecionada: m_val = op5_es / 100
            elif "Opção 6" in meta_selecionada: m_val = op6_es / 100
            elif "Opção 4" in meta_selecionada: m_val = rcv # Absoluto
            else: m_val = custom_val

            if "Opção 4" in meta_selecionada:
                df_final['LSC_Clin'] = mu_robusto + m_val
                df_final['LIC_Clin'] = mu_robusto - m_val
            else:
                df_final['LSC_Clin'] = mu_robusto * (1 + m_val)
                df_final['LIC_Clin'] = mu_robusto * (1 - m_val)

            # Limite Estatístico DP Móvel (Sua fórmula do Fator t)
            fator_t = stats.t.ppf(0.99999, n_janela-1)
            df_final['Limite_DP_Est'] = fator_t * (0.75 * cva)

            # --- RENDERIZAÇÃO COM PLOTLY WEBGL ---
            
            # 1. Gráfico Mediana Móvel
            fig_med = go.Figure()
            fig_med.add_trace(go.Scattergl(x=df_final['timestamp'], y=df_final['Mediana_Movel'], name="Mediana Móvel", line=dict(color='blue')))
            # Limites Estatísticos (Estabilidade)
            fig_med.add_trace(go.Scattergl(x=df_final['timestamp'], y=df_final['LSC_Est'], name="LSC Estatístico", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df_final['timestamp'], y=df_final['LIC_Est'], name="LIC Estatístico", line=dict(color='orange', dash='dot')))
            # Limites Clínicos (Qualidade Analítica)
            fig_med.add_trace(go.Scattergl(x=df_final['timestamp'], y=df_final['LSC_Clin'], name="LSC Clínico", line=dict(color='red')))
            fig_med.add_trace(go.Scattergl(x=df_final['timestamp'], y=df_final['LIC_Clin'], name="LIC Clínico", line=dict(color='red')))
            
            fig_med.update_layout(title="Mediana Móvel: Limites Estatísticos (ISO) vs Clínicos (VB)", hovermode="x unified", height=500)
            st.plotly_chart(fig_med, use_container_width=True)

            # 2. Gráfico DP Móvel
            fig_dp = go.Figure()
            fig_dp.add_trace(go.Scattergl(x=df_final['timestamp'], y=df_final['DP_Movel'], name="DP Móvel", line=dict(color='purple')))
            fig_dp.add_trace(go.Scattergl(x=df_final['timestamp'], y=df_final['Limite_DP_Est'], name="Limite Superior (t-99.999%)", line=dict(color='red', dash='dash')))
            
            fig_dp.update_layout(title="Precisão: Desvio Padrão Móvel", hovermode="x unified", height=400)
            st.plotly_chart(fig_dp, use_container_width=True)

            # --- SUMÁRIO ---
            st.divider()
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Média Robusta (ISO)", f"{mu_robusto:.4f}")
            c_m2.metric("DP Robusto (ISO)", f"{s_robusto:.4f}")
            c_m3.metric("N (Janela)", n_janela)

            # Identificação de Violações
            viola_est = df_final[(df_final['Mediana_Movel'] > df_final['LSC_Est']) | (df_final['Mediana_Movel'] < df_final['LIC_Est'])]
            viola_clin = df_final[(df_final['Mediana_Movel'] > df_final['LSC_Clin']) | (df_final['Mediana_Movel'] < df_final['LIC_Clin'])]
            
            if not viola_clin.empty:
                st.error(f"CUIDADO: Foram detectadas {len(viola_clin)} violações de Limite Clínico!")
                st.dataframe(viola_clin[[c_data, c_hora, c_res, 'Mediana_Movel']].head(100))
