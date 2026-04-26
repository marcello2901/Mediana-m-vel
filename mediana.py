import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Configuração de Layout e Performance para suportar grandes volumes de dados
st.set_page_config(page_title="Medida Móvel - Ferramenta de Gestão Analítica", layout="wide")

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

# --- BLOCO 2: PROCESSAMENTO DE DADOS ---

@st.cache_data
def processar_estatistica_avancada(df_raw, col_res, col_data, col_hora, n_janela):
    """Realiza a limpeza, ordenação cronológica e cálculos móveis."""
    df = df_raw.copy()
    
    # Tratamento de dados numéricos (converte vírgula brasileira para ponto)
    df[col_res] = pd.to_numeric(df[col_res].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
    df = df.dropna(subset=[col_res])
    
    # Ordenação Cronológica Estrita por Data e Hora
    df['timestamp'] = pd.to_datetime(df[col_data].astype(str) + ' ' + df[col_hora].astype(str), dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 1. Parâmetros Robustos Globais (ISO 13528)
    mu_r, s_r = algoritmo_a_robusto(df[col_res])
    
    # 2. Cálculos Móveis
    # Nota: Rolling median e std para monitoramento da tendência e imprecisão
    df['Mediana_Movel'] = df[col_res].rolling(window=int(n_janela), min_periods=1).median()
    df['DP_Movel'] = df[col_res].rolling(window=int(n_janela), min_periods=2).std().fillna(0)
    
    # 3. Limites Estatísticos (Variação inerente do equipamento)
    # 1.2533 é o ajuste do erro padrão para a mediana
    erro_padrao_mediana = (1.2533 * s_r) / np.sqrt(n_janela)
    df['LSC_Est'] = mu_r + (3 * erro_padrao_mediana)
    df['LIC_Est'] = mu_r - (3 * erro_padrao_mediana)
    
    # N Total para os cálculos de meta baseados em Graus de Liberdade
    n_total = len(df)
    
    return df, mu_r, s_r, n_total

# --- BLOCO 3: INTERFACE ---

st.title("📊 Ferramenta Medida Móvel - Performance Analítica")

with st.sidebar:
    st.header("⚙️ Especificações da Qualidade")
    cvi = st.number_input("CVi (%)", value=2.0, step=0.1)
    cvg = st.number_input("CVg (%)", value=5.0, step=0.1)
    cva = st.number_input("CVa (Analítico) (%)", value=3.0, step=0.1)
    
    st.divider()
    
    # SEÇÃO: Erro Aleatório Máximo Escolhido (Define o limite do DP Móvel)
    st.subheader("Configuração: Erro Aleatório (EA)")
    
    # Placeholder para cálculo dinâmico baseado no N do arquivo
    n_total_placeholder = st.empty()
    
    ea_escolhido = st.selectbox("EA Máximo Escolhido:", [
        "Opção 1 - EA Desejável",
        "Opção 2 - EA Mínima",
        "Opção 3 - Manual (%)"
    ])
    
    st.divider()

    # SEÇÃO: Erro Sistemático Máximo Escolhido (Define os limites da Mediana)
    st.subheader("Configuração: Erro Sistemático (ES)")
    
    es_escolhido = st.selectbox("ES Máximo Escolhido:", [
        "Opção 5 - ES Desejável",
        "Opção 6 - ES Mínima",
        "Opção 4 - RCV (Absoluto)",
        "Opção 7 - Manual (%)"
    ])

    n_janela = st.number_input("Tamanho da Janela (N Móvel)", value=20, min_value=2)

# Fluxo de Upload
uploaded_file = st.file_uploader("Suba sua planilha de resultados", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    st.info(f"Registros carregados: {len(df_raw)}")
    
    cols = df_raw.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1: c_data = st.selectbox("Data", cols)
    with c2: c_hora = st.selectbox("Hora", cols)
    with c3: c_res = st.selectbox("Resultado", cols)

    if st.button("🚀 Executar Análise"):
        with st.spinner("Processando dados e aplicando algoritmos robustos..."):
            df, mu_r, s_r, n_total = processar_estatistica_avancada(df_raw, c_res, c_data, c_hora, n_janela)
            
            # --- CÁLCULO DAS METAS CONFORME SUAS FÓRMULAS ---
            # Graus de Liberdade (E57) = N Total - 1
            df_degrees = n_total - 1
            t_factor = stats.t.ppf(0.99999, df_degrees)
            
            # Opção 1: EA Desejável = INV.T(0.99999; df) * (0.75 * CVa)
            ea_desejavel = t_factor * (0.75 * cva)
            # Opção 2: EA Mínimo = INV.T(0.99999; df) * (0.5 * CVa)
            ea_minimo = t_factor * (0.5 * cva)
            
            # ES Baseado em Variação Biológica
            es_desejavel = 0.25 * np.sqrt(cvi**2 + cvg**2)
            es_minimo = 0.375 * np.sqrt(cvi**2 + cvg**2)
            rcv_val = 2.77 * np.sqrt(cva**2 + cvi**2)

            # --- DEFINIÇÃO DOS VALORES SELECIONADOS ---
            # EA (Para o DP Móvel)
            if "Opção 1" in ea_escolhido: val_ea = ea_desejavel / 100
            elif "Opção 2" in ea_escolhido: val_ea = ea_minimo / 100
            else: val_ea = st.sidebar.number_input("EA Manual (%)", value=5.0) / 100

            # ES (Para a Mediana Móvel)
            if "Opção 5" in es_escolhido: val_es = es_desejavel / 100
            elif "Opção 6" in es_escolhido: val_es = es_minimo / 100
            elif "Opção 4" in es_escolhido: val_es = rcv_val
            else: val_es = st.sidebar.number_input("ES Manual (%)", value=5.0) / 100

            # --- APLICAÇÃO DOS LIMITES NOS GRÁFICOS ---
            
            # 1. Mediana Móvel (Exatidão / Erro Sistemático)
            if "Opção 4" in es_escolhido:
                df['LSC_Clin_ES'] = mu_r + val_es
                df['LIC_Clin_ES'] = mu_r - val_es
            else:
                df['LSC_Clin_ES'] = mu_r * (1 + val_es)
                df['LIC_Clin_ES'] = mu_r * (1 - val_es)

            # 2. DP Móvel (Precisão / Erro Aleatório)
            df['LSC_DP_Clin_EA'] = mu_r * val_ea
            
            # Limite Estatístico do DP (Incerteza técnica)
            fator_t_est = stats.t.ppf(0.99999, n_janela - 1)
            df['LSC_DP_Est'] = fator_t_est * (0.75 * (cva/100) * mu_r)

            # --- RENDERIZAÇÃO ---
            
            st.subheader("Gráfico: Controle de Exatidão (Mediana Móvel)")
            fig_med = go.Figure()
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['Mediana_Movel'], name="Mediana Móvel", line=dict(color='#1f77b4', width=2)))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_Est'], name="LSC Estatístico", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LIC_Est'], name="LIC Estatístico", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_Clin_ES'], name="LSC Clínico (ES)", line=dict(color='red', width=1.5)))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LIC_Clin_ES'], name="LIC Clínico (ES)", line=dict(color='red', width=1.5)))
            fig_med.update_layout(hovermode="x unified", template="plotly_white", height=500)
            st.plotly_chart(fig_med, use_container_width=True)

            st.subheader("Gráfico: Controle de Precisão (DP Móvel)")
            fig_dp = go.Figure()
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['DP_Movel'], name="DP Móvel", line=dict(color='purple', width=2)))
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_DP_Est'], name="LSC Estatístico", line=dict(color='orange', dash='dash')))
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_DP_Clin_EA'], name="LSC Clínico (EA)", line=dict(color='red', width=2.5)))
            fig_dp.update_layout(hovermode="x unified", template="plotly_white", height=400)
            st.plotly_chart(fig_dp, use_container_width=True)

            # Sumário de Metas Calculadas
            st.divider()
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric("EA Desejável Calculado", f"{ea_desejavel:.2f}%")
            col_met2.metric("EA Mínimo Calculado", f"{ea_minimo:.2f}%")
            col_met3.metric("Média Robusta (Target)", f"{mu_r:.4f}")
