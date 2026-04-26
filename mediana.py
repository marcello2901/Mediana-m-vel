import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Configuração de Layout e Performance
st.set_page_config(page_title="Medida Móvel - Gestão de Qualidade Analítica", layout="wide")

# --- BLOCO 1: ALGORITMOS ROBUSTOS (ISO 13528) ---

@st.cache_data
def algoritmo_a_robusto(serie, max_iter=25):
    """
    Implementação do Algoritmo A (ISO 13528) para Média e DP Robustos.
    Define a linha de base (Target) e a variabilidade técnica do equipamento.
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

# --- BLOCO 2: PROCESSAMENTO E CÁLCULOS MÓVEIS ---

@st.cache_data
def processar_dados_completos(df_raw, col_res, col_data, col_hora, n_janela):
    """Limpeza, ordenação e cálculo das estatísticas móveis."""
    df = df_raw.copy()
    
    # Tratamento de decimais (vírgula para ponto)
    df[col_res] = pd.to_numeric(df[col_res].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
    df = df.dropna(subset=[col_res])
    
    # Ordenação Cronológica
    df['timestamp'] = pd.to_datetime(df[col_data].astype(str) + ' ' + df[col_hora].astype(str), dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Média e DP Robusto (ISO 13528)
    mu_r, s_r = algoritmo_a_robusto(df[col_res])
    
    # Mediana e DP Móvel
    df['Mediana_Movel'] = df[col_res].rolling(window=int(n_janela), min_periods=1).median()
    df['DP_Movel'] = df[col_res].rolling(window=int(n_janela), min_periods=2).std().fillna(0)
    
    # Limites Estatísticos (Variação inerente do processo)
    # 1.2533 é o fator de correção para erro padrão da mediana
    erro_pad_mediana = (1.2533 * s_r) / np.sqrt(n_janela)
    df['LSC_Est'] = mu_r + (3 * erro_pad_mediana)
    df['LIC_Est'] = mu_r - (3 * erro_pad_mediana)
    
    return df, mu_r, s_r

# --- BLOCO 3: INTERFACE DO USUÁRIO ---

st.title("📊 Ferramenta de Medida Móvel - Gestão de Erros EA/ES")

with st.sidebar:
    st.header("⚙️ Especificações da Qualidade")
    cvi = st.number_input("CVi (%)", value=2.0, step=0.1)
    cvg = st.number_input("CVg (%)", value=5.0, step=0.1)
    cva = st.number_input("Pior Cenário CVa (%)", value=3.0, step=0.1)
    n_janela = st.number_input("N (Tamanho da Janela)", value=50, min_value=2)
    
    st.divider()

    # --- CÁLCULO DAS METAS (Conforme Planilha do Marcello) ---
    # Fator de confiança t-Student para o N escolhido (95% bicaudal -> 0.975 ou 99% bicaudal -> 0.995)
    fator_t_meta = stats.t.ppf(0.995, n_janela - 1) 
    
    # EA baseia-se no CVa e no fator estatístico da Mediana
    ea_desejavel = fator_t_meta * 0.8 * cva
    ea_minimo = ea_desejavel / 1.5
    
    # ES baseia-se na Variação Biológica (Milan/Fraser)
    es_desejavel = 0.25 * np.sqrt(cvi**2 + cvg**2)
    es_minimo = 0.375 * np.sqrt(cvi**2 + cvg**2)
    rcv_fator = ea_desejavel * 2.66 # RCV atrelado ao Erro Aleatório

    # SEÇÃO: Erro Aleatório Máximo Escolhido (Define o Limite do DP Móvel)
    st.subheader("Configuração: Erro Aleatório (EA)")
    ea_selecionado = st.selectbox("EA Máximo Escolhido (Precisão):", [
        f"Opção 1 - EA Desejável ({ea_desejavel:.2f}%)",
        f"Opção 2 - EA Mínima ({ea_minimo:.2f}%)",
        "Opção 3 - Manual (%)"
    ])
    
    if "Opção 1" in ea_selecionado: val_ea = ea_desejavel / 100
    elif "Opção 2" in ea_selecionado: val_ea = ea_minimo / 100
    else: val_ea = st.number_input("EA Manual (%)", value=5.0) / 100

    st.divider()

    # SEÇÃO: Erro Sistemático Máximo Escolhido (Define os Limites da Mediana Móvel)
    st.subheader("Configuração: Erro Sistemático (ES)")
    es_selecionado = st.selectbox("ES Máximo Escolhido (Exatidão):", [
        f"Opção 5 - ES Desejável ({es_desejavel:.2f}%)",
        f"Opção 6 - ES Mínima ({es_minimo:.2f}%)",
        f"Opção 4 - RCV ({rcv_fator:.2f} Absoluto)",
        "Opção 7 - Manual (%)"
    ])

    if "Opção 5" in es_selecionado: val_es = es_desejavel / 100
    elif "Opção 6" in es_selecionado: val_es = es_minimo / 100
    elif "Opção 4" in es_selecionado: val_es = rcv_fator / 100 # Tratado como percentual do mu
    else: val_es = st.number_input("ES Manual (%)", value=5.0) / 100

# --- CARREGAMENTO E EXECUÇÃO ---

uploaded_file = st.file_uploader("Suba sua planilha de resultados", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        df_input = pd.read_excel(uploaded_file)
    
    st.info(f"Registros carregados: {len(df_input)}")
    cols = df_input.columns.tolist()
    
    c1, c2, c3 = st.columns(3)
    with c1: col_d = st.selectbox("Data", cols)
    with c2: col_h = st.selectbox("Hora", cols)
    with c3: col_r = st.selectbox("Resultado", cols)

    if st.button("🚀 Processar Análise Analítica"):
        with st.spinner("Calculando Limites ISO e Metas de Qualidade..."):
            df, mu_r, s_r = processar_dados_completos(df_input, col_r, col_d, col_h, n_janela)
            
            # 1. LIMITES DA MEDIANA (Baseados no Erro Sistemático - ES)
            df['LSC_Mediana_Clin'] = mu_r * (1 + val_es)
            df['LIC_Mediana_Clin'] = mu_r * (1 - val_es)

            # 2. LIMITES DO DP (Baseados no Erro Aleatório - EA)
            df['LSC_DP_Clin'] = mu_r * val_ea
            
            # Limite Estatístico do DP (t-Student 99.999% da planilha original)
            fator_t_est = stats.t.ppf(0.99999, n_janela - 1)
            df['LSC_DP_Est'] = fator_t_est * (0.75 * (cva/100) * mu_r)

            # --- GRÁFICOS (PLOTLY WEBGL) ---

            # Gráfico 1: Mediana Móvel (Exatidão)
            fig_med = go.Figure()
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['Mediana_Movel'], name="Mediana Móvel", line=dict(color='#1f77b4', width=2)))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_Est'], name="LSC Estatístico", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LIC_Est'], name="LIC Estatístico", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_Mediana_Clin'], name="LSC Clínico (ES)", line=dict(color='red', width=2)))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LIC_Mediana_Clin'], name="LIC Clínico (ES)", line=dict(color='red', width=2)))
            fig_med.update_layout(title="Controle de Mediana Móvel (Exatidão / Erro Sistemático)", hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_med, use_container_width=True)

            # Gráfico 2: DP Móvel (Precisão)
            fig_dp = go.Figure()
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['DP_Movel'], name="DP Móvel", line=dict(color='purple', width=2)))
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_DP_Est'], name="LSC Estatístico (Incerteza)", line=dict(color='orange', dash='dash')))
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_DP_Clin'], name="LSC Clínico (EA)", line=dict(color='red', width=2.5)))
            fig_dp.update_layout(title="Controle de DP Móvel (Precisão / Erro Aleatório)", hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_dp, use_container_width=True)

            # --- SUMÁRIO ---
            st.success(f"Análise Concluída para N={n_janela}. Target Robusto: {mu_r:.4f}")
            c_a1, c_a2 = st.columns(2)
            with c_a1:
                st.metric("Meta ES Escolhida", f"{val_es*100:.2f} %")
            with c_a2:
                st.metric("Meta EA Escolhida", f"{val_ea*100:.2f} %")
