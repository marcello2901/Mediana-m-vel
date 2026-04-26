import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Configuração de Layout e Performance para suportar grandes volumes de dados
st.set_page_config(page_title="Medida Móvel - Gestão de Erros Analíticos", layout="wide")

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
    # Nota: Rolling median e std para monitoramento em tempo real
    df['Mediana_Movel'] = df[col_res].rolling(window=int(n_janela), min_periods=1).median()
    df['DP_Movel'] = df[col_res].rolling(window=int(n_janela), min_periods=2).std().fillna(0)
    
    # 3. Limites Estatísticos (Faixa de estabilidade técnica do equipamento)
    # 1.2533 é o ajuste do erro padrão para a mediana
    erro_padrao_mediana = (1.2533 * s_r) / np.sqrt(n_janela)
    df['LSC_Est'] = mu_r + (3 * erro_padrao_mediana)
    df['LIC_Est'] = mu_r - (3 * erro_padrao_mediana)
    
    return df, mu_r, s_r

# --- BLOCO 3: INTERFACE ---

st.title("📊 Ferramenta Medida Móvel - Gestão de Erros EA e ES")

with st.sidebar:
    st.header("⚙️ Especificações da Qualidade")
    cvi = st.number_input("CVi (%)", value=2.0, step=0.1) / 100
    cvg = st.number_input("CVg (%)", value=5.0, step=0.1) / 100
    cva = st.number_input("Pior Cenário CVa (%)", value=3.0, step=0.1) / 100
    
    st.divider()
    
    # Cálculos Prévios baseados na Variação Biológica
    ea_desejavel = 0.5 * cvi
    ea_minimo = 0.25 * cvi
    es_desejavel = 0.25 * np.sqrt(cvi**2 + cvg**2)
    es_minimo = 0.375 * np.sqrt(cvi**2 + cvg**2)
    rcv_val = 2.77 * np.sqrt(cva**2 + cvi**2)

    # SEÇÃO: Erro Aleatório Máximo Escolhido (Define o limite do DP Móvel)
    st.subheader("Precisão")
    ea_escolhido = st.selectbox("Erro Aleatório Máximo Escolhido (EA):", [
        f"Opção 1 - EA Desejável ({ea_desejavel*100:.2f}%)",
        f"Opção 2 - EA Mínima ({ea_minimo*100:.2f}%)",
        "Opção 3 - Outra Fonte (Manual %)"
    ])
    
    if "Opção 1" in ea_escolhido: val_ea = ea_desejavel
    elif "Opção 2" in ea_escolhido: val_ea = ea_minimo
    else: val_ea = st.number_input("EA Manual (%)", value=5.0) / 100

    st.divider()

    # SEÇÃO: Erro Sistemático Máximo Escolhido (Define o limite da Mediana Móvel)
    st.subheader("Exatidão")
    es_escolhido = st.selectbox("Erro Sistemático Máximo Escolhido (ES):", [
        f"Opção 5 - ES Desejável ({es_desejavel*100:.2f}%)",
        f"Opção 6 - ES Mínima ({es_minimo*100:.2f}%)",
        f"Opção 4 - RCV ({rcv_val:.2f} Absoluto)",
        "Opção 7 - Outra Fonte (Manual %)"
    ])

    if "Opção 5" in es_escolhido: val_es = es_desejavel
    elif "Opção 6" in es_escolhido: val_es = es_minimo
    elif "Opção 4" in es_escolhido: val_es = rcv_val
    else: val_es = st.number_input("ES Manual (%)", value=5.0) / 100

    n_janela = st.number_input("N (Tamanho da Janela)", value=20, min_value=2)

# Fluxo de Upload
uploaded_file = st.file_uploader("Suba sua planilha (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    st.info(f"Registros carregados: {len(df_raw)}")
    
    cols = df_raw.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1: c_data = st.selectbox("Coluna Data", cols)
    with c2: c_hora = st.selectbox("Coluna Hora", cols)
    with c3: c_res = st.selectbox("Coluna Resultado", cols)

    if st.button("🚀 Processar Análise"):
        with st.spinner("Executando cálculos robustos e gerando gráficos..."):
            df, mu_r, s_r = processar_estatistica_avancada(df_raw, c_res, c_data, c_hora, n_janela)
            
            # 1. CÁLCULO LIMITES CLÍNICOS DA MEDIANA (Baseado no ES)
            if "Opção 4" in es_escolhido:
                df['LSC_Clin_ES'] = mu_r + val_es
                df['LIC_Clin_ES'] = mu_r - val_es
            else:
                df['LSC_Clin_ES'] = mu_r * (1 + val_es)
                df['LIC_Clin_ES'] = mu_r * (1 - val_es)

            # 2. CÁLCULO LIMITE CLÍNICO DO DP (Baseado no EA)
            # O limite clínico de imprecisão é o alvo robusto multiplicado pelo erro aleatório tolerado
            df['LSC_DP_Clin_EA'] = mu_r * val_ea
            
            # Limite Estatístico do DP (Usando sua fórmula com t-99.999%)
            fator_t = stats.t.ppf(0.99999, n_janela-1)
            df['LSC_DP_Est'] = fator_t * (0.75 * cva)

            # --- GRÁFICOS WEBGL (PLOTLY) ---
            
            # Gráfico de Mediana Móvel
            fig_med = go.Figure()
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['Mediana_Movel'], name="Mediana Móvel", line=dict(color='#1f77b4', width=2.5)))
            # Limites Estatísticos
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_Est'], name="LSC Estatístico", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LIC_Est'], name="LIC Estatístico", line=dict(color='orange', dash='dot')))
            # Limites Clínicos de ES
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_Clin_ES'], name="LSC Clínico (Erro Sistemático)", line=dict(color='red', width=1.5)))
            fig_med.add_trace(go.Scattergl(x=df['timestamp'], y=df['LIC_Clin_ES'], name="LIC Clínico (Erro Sistemático)", line=dict(color='red', width=1.5)))
            
            fig_med.update_layout(title="Mediana Móvel (Controle de Exatidão)", hovermode="x unified", height=500, template="plotly_white")
            st.plotly_chart(fig_med, use_container_width=True)

            # Gráfico de DP Móvel
            fig_dp = go.Figure()
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['DP_Movel'], name="DP Móvel", line=dict(color='purple', width=2)))
            # Limite Estatístico
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_DP_Est'], name="LSC Estatístico (t-Student)", line=dict(color='orange', dash='dash')))
            # Limite Clínico de EA
            fig_dp.add_trace(go.Scattergl(x=df['timestamp'], y=df['LSC_DP_Clin_EA'], name="LSC Clínico (Erro Aleatório)", line=dict(color='red', width=2.5)))
            
            fig_dp.update_layout(title="DP Móvel (Controle de Precisão)", hovermode="x unified", height=400, template="plotly_white")
            st.plotly_chart(fig_dp, use_container_width=True)

            # --- SUMÁRIO TÉCNICO ---
            st.divider()
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Média Robusta (ISO 13528)", f"{mu_r:.4f}")
            c_m2.metric("DP Robusto (ISO 13528)", f"{s_r:.4f}")
            c_m3.metric("N (Janela)", n_janela)

            # Alertas Clínicos
            violas = df[(df['Mediana_Movel'] > df['LSC_Clin_ES']) | (df['Mediana_Movel'] < df['LIC_Clin_ES']) | (df['DP_Movel'] > df['LSC_DP_Clin_EA'])]
            if not violas.empty:
                st.error(f"🚨 {len(violas)} violações de limite clínico detectadas!")
                st.dataframe(violas[[c_data, c_hora, c_res, 'Mediana_Movel', 'DP_Movel']].head(100))
