import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Configuração de Layout da Página e Performance
st.set_page_config(
    page_title="Gestão de Medida Móvel - Qualidade Analítica",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BLOCO 1: ALGORITMOS ROBUSTOS (CONFORME ISO 13528:2005) ---

@st.cache_data
def algoritmo_a_robusto(serie, max_iter=25):
    """
    Implementação do Algoritmo A para Média Robusta (mu) e DP Robusto (s).
    Este algoritmo elimina o viés de outliers de forma iterativa.
    """
    x = serie.values
    mu = np.median(x)
    s = 1.483 * np.median(np.abs(x - mu))
    
    for _ in range(max_iter):
        delta = 1.5 * s
        # Limita os valores entre mu - delta e mu + delta
        x_cap = np.clip(x, mu - delta, mu + delta)
        mu_new = np.mean(x_cap)
        s_new = 1.134 * np.std(x_cap, ddof=1)
        
        # Critério de convergência
        if abs(mu - mu_new) < 1e-6 and abs(s - s_new) < 1e-6:
            break
        mu, s = mu_new, s_new
        
    return mu, s

def tratar_outliers_tukey(serie):
    """Substitui outliers (1.5*IQR) pela mediana para suavizar a Mediana Móvel."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    return serie.mask((serie < lim_inf) | (serie > lim_sup), serie.median())

# --- BLOCO 2: PROCESSAMENTO E CÁLCULOS TÉCNICOS ---

@st.cache_data
def processar_analise_completa(df_raw, col_res, col_data, col_hora, n_janela):
    """Realiza a limpeza, ordenação cronológica e gera as séries móveis."""
    df = df_raw.copy()
    
    # Tratamento de dados (vírgula por ponto e remoção de nulos)
    df[col_res] = pd.to_numeric(df[col_res].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
    df = df.dropna(subset=[col_res])
    
    # Combinação de Data e Hora para ordenação correta
    df['timestamp'] = pd.to_datetime(df[col_data].astype(str) + ' ' + df[col_hora].astype(str), dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Média e DP Robusto Globais (Alvo do Equipamento)
    mu_robusto, s_robusto = algoritmo_a_robusto(df[col_res])
    
    # Dados limpos de outliers extremos (Tukey) para a Mediana Móvel
    df['res_limpo'] = tratar_outliers_tukey(df[col_res])
    
    # Cálculos Móveis
    df['Mediana_Movel'] = df['res_limpo'].rolling(window=int(n_janela), min_periods=1).median()
    df['DP_Movel'] = df[col_res].rolling(window=int(n_janela), min_periods=2).std().fillna(0)
    
    # Limites Estatísticos (Variação Técnica do Instrumento)
    # 1.2533 é o fator de erro padrão para medianas em distribuições normais
    erro_padrao_mediana = (1.2533 * s_robusto) / np.sqrt(n_janela)
    df['LSC_Est'] = mu_robusto + (3 * erro_padrao_mediana)
    df['LIC_Est'] = mu_robusto - (3 * erro_padrao_mediana)
    
    return df, mu_robusto, s_robusto, len(df)

# --- BLOCO 3: INTERFACE PRINCIPAL ---

st.title("📊 Ferramenta Medida Móvel - Qualidade de Laboratório")

with st.sidebar:
    st.header("⚙️ Parâmetros Analíticos")
    cvi = st.number_input("CVi (Variação Individual %)", value=2.0, step=0.1)
    cvg = st.number_input("CVg (Variação Grupo %)", value=5.0, step=0.1)
    cva = st.number_input("CVa (Pior Cenário Analítico %)", value=3.0, step=0.1)
    
    st.divider()
    
    st.subheader("Configuração: Erro Aleatório (EA)")
    ea_opcao = st.selectbox("EA Máximo Escolhido (Precisão):", 
                             ["Opção 1 - EA Desejável", "Opção 2 - EA Mínima", "Opção 3 - Manual (%)"])
    
    st.divider()
    
    st.subheader("Configuração: Erro Sistemático (ES)")
    es_opcao = st.selectbox("ES Máximo Escolhido (Exatidão):", 
                             ["Opção 5 - ES Desejável", "Opção 6 - ES Mínima", "Opção 4 - RCV (Absoluto)", "Opção 7 - Manual (%)"])

    n_janela = st.number_input("Tamanho da Janela Móvel (N)", value=50, min_value=2)

# Upload do Arquivo
file = st.file_uploader("Selecione a planilha de resultados (CSV ou XLSX)", type=["csv", "xlsx"])

if file:
    # Leitura do arquivo
    if file.name.endswith('.csv'):
        df_input = pd.read_csv(file, sep=None, engine='python')
    else:
        df_input = pd.read_excel(file)
    
    st.write(f"Total de registros carregados: {len(df_input)}")
    
    # Seleção de Colunas
    cols_planilha = df_input.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1: col_data = st.selectbox("Coluna de Data", cols_planilha)
    with c2: col_hora = st.selectbox("Coluna de Hora", cols_planilha)
    with c3: col_resultado = st.selectbox("Coluna de Resultado", cols_planilha)

    if st.button("🚀 Processar e Gerar Gráficos"):
        with st.spinner("Processando algoritmos robustos e metas analíticas..."):
            
            # Processamento Principal
            df_res, mu_r, s_r, n_total_estudo = processar_analise_completa(df_input, col_resultado, col_data, col_hora, n_janela)
            
            # --- CÁLCULO DAS METAS (CONFORME FÓRMULAS DA PLANILHA) ---
            # Graus de Liberdade = N total do arquivo - 1
            df_degrees = n_total_estudo - 1
            t_student = stats.t.ppf(0.99999, df_degrees)
            
            # Metas de Erro Aleatório (EA)
            ea_calc_desejavel = t_student * (0.75 * cva)
            ea_calc_minimo = t_student * (0.5 * cva)
            
            # Metas de Erro Sistemático (ES) baseadas em Variação Biológica
            es_calc_desejavel = 0.25 * np.sqrt(cvi**2 + cvg**2)
            es_calc_minimo = 0.375 * np.sqrt(cvi**2 + cvg**2)
            rcv_calc = 2.77 * np.sqrt(cva**2 + cvi**2)

            # --- PAINEL DE METAS CALCULADAS (MÉTRICAS) ---
            st.subheader("🎯 Painel de Metas Calculadas")
            m1, m2, m3, m4 = st.columns(4)
            
            # Seleção de EA para aplicação e visualização
            if "Opção 1" in ea_opcao:
                val_ea_final = ea_calc_desejavel
                m1.metric("EA Desejável (Alvo)", f"{ea_calc_desejavel:.2f}%")
            elif "Opção 2" in ea_opcao:
                val_ea_final = ea_calc_minimo
                m1.metric("EA Mínimo (Alvo)", f"{ea_calc_minimo:.2f}%")
            else:
                val_ea_final = st.sidebar.number_input("Valor EA Manual (%)", value=5.0)
                m1.metric("EA Manual", f"{val_ea_final:.2f}%")

            # Seleção de ES para aplicação e visualização
            if "Opção 5" in es_opcao:
                val_es_final = es_calc_desejavel
                m2.metric("ES Desejável (Alvo)", f"{es_calc_desejavel:.2f}%")
            elif "Opção 6" in es_opcao:
                val_es_final = es_calc_minimo
                m2.metric("ES Mínimo (Alvo)", f"{es_calc_minimo:.2f}%")
            elif "Opção 4" in es_opcao:
                val_es_final = rcv_calc
                m2.metric("RCV Calculado (Absoluto)", f"{rcv_calc:.2f}")
            else:
                val_es_final = st.sidebar.number_input("Valor ES Manual (%)", value=5.0)
                m2.metric("ES Manual", f"{val_es_final:.2f}%")

            m3.metric("Média Robusta (Target)", f"{mu_r:.4f}")
            m4.metric("N Total Estudo (GL)", n_total_estudo)
            
            st.divider()

            # --- APLICAÇÃO DOS LIMITES NOS GRÁFICOS ---
            
            # 1. Limites Mediana Móvel (Exatidão / ES)
            if "Opção 4" in es_opcao:
                df_res['LSC_Clin_ES'] = mu_r + val_es_final
                df_res['LIC_Clin_ES'] = mu_r - val_es_final
            else:
                df_res['LSC_Clin_ES'] = mu_r * (1 + (val_es_final / 100))
                df_res['LIC_Clin_ES'] = mu_r * (1 - (val_es_final / 100))

            # 2. Limites DP Móvel (Precisão / EA)
            # Limite Clínico: Média Robusta * EA escolhido
            df_res['LSC_DP_Clin_EA'] = mu_r * (val_ea_final / 100)
            
            # Limite Estatístico DP: Fator t para a Janela Móvel (N-1)
            t_est_janela = stats.t.ppf(0.99999, n_janela - 1)
            df_res['LSC_DP_Est'] = t_est_janela * (0.75 * (cva/100) * mu_r)

            # --- CONSTRUÇÃO DOS GRÁFICOS (PLOTLY WEBGL) ---

            # Gráfico de Mediana Móvel (Exatidão)
            fig_med = go.Figure()
            fig_med.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['Mediana_Movel'], name="Mediana Móvel", line=dict(color='#1f77b4', width=2)))
            fig_med.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['LSC_Est'], name="LSC Estatístico (Equipamento)", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['LIC_Est'], name="LIC Estatístico (Equipamento)", line=dict(color='orange', dash='dot')))
            fig_med.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['LSC_Clin_ES'], name="LSC Clínico (Erro Sistemático)", line=dict(color='red', width=2)))
            fig_med.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['LIC_Clin_ES'], name="LIC Clínico (Erro Sistemático)", line=dict(color='red', width=2)))
            fig_med.update_layout(title="Controle de Mediana Móvel (Exatidão)", hovermode="x unified", template="plotly_white", height=500)
            st.plotly_chart(fig_med, use_container_width=True)

            # Gráfico de DP Móvel (Precisão)
            fig_dp = go.Figure()
            fig_dp.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['DP_Movel'], name="DP Móvel", line=dict(color='purple', width=2)))
            fig_dp.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['LSC_DP_Est'], name="LSC Estatístico (Incerteza)", line=dict(color='orange', dash='dash')))
            fig_dp.add_trace(go.Scattergl(x=df_res['timestamp'], y=df_res['LSC_DP_Clin_EA'], name="LSC Clínico (Erro Aleatório)", line=dict(color='red', width=2.5)))
            fig_dp.update_layout(title="Controle de DP Móvel (Precisão)", hovermode="x unified", template="plotly_white", height=400)
            st.plotly_chart(fig_dp, use_container_width=True)

            # Tabela de Alertas de Violação
            alertas = df_res[(df_res['Mediana_Movel'] > df_res['LSC_Clin_ES']) | 
                             (df_res['Mediana_Movel'] < df_res['LIC_Clin_ES']) | 
                             (df_res['DP_Movel'] > df_res['LSC_DP_Clin_EA'])]
            
            if not alertas.empty:
                st.error(f"🚨 {len(alertas)} violações de Limite Clínico detectadas!")
                st.dataframe(alertas[[col_data, col_hora, col_resultado, 'Mediana_Movel', 'DP_Movel']].head(100))
