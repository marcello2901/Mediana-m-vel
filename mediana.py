import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# Configuração de performance e layout
st.set_page_config(page_title="Medida Móvel - Alta Performance", layout="wide")

# --- FUNÇÕES DE CÁLCULO COM CACHE ---

@st.cache_data
def processar_dados_robustos(df_raw, col_res, col_data, col_hora, n_janela):
    """Processa a limpeza, ordenação e cálculos móveis com alta eficiência."""
    df = df_raw.copy()
    
    # Conversão numérica brasileira (vírgula para ponto)
    df[col_res] = df[col_res].astype(str).str.replace(',', '.').str.strip()
    df[col_res] = pd.to_numeric(df[col_res], errors='coerce')
    df = df.dropna(subset=[col_res])
    
    # Ordenação Cronológica
    df['timestamp'] = pd.to_datetime(df[col_data].astype(str) + ' ' + df[col_hora].astype(str), dayfirst=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Tratamento de Outliers (Tukey)
    q1 = df[col_res].quantile(0.25)
    q3 = df[col_res].quantile(0.75)
    iqr = q3 - q1
    mediana_geral = df[col_res].median()
    df['res_limpo'] = df[col_res].mask(
        (df[col_res] < (q1 - 1.5 * iqr)) | (df[col_res] > (q3 + 1.5 * iqr)), 
        mediana_geral
    )
    
    # Cálculos Móveis
    df['Mediana_Movel'] = df['res_limpo'].rolling(window=int(n_janela), min_periods=1).median()
    df['DP_Movel'] = df['res_limpo'].rolling(window=int(n_janela), min_periods=2).std().fillna(0)
    
    return df

def criar_grafico_webgl(df, y_col, lsc_col, lic_col, titulo, y_label):
    """Gera gráficos usando WebGL para suportar milhares de pontos sem travar."""
    fig = go.Figure()
    
    # Linha Principal (Mediana ou DP)
    fig.add_trace(go.Scattergl(x=df['timestamp'], y=df[y_col], name=y_label,
                               line=dict(color='#1f77b4', width=2)))
    
    # Limite Superior
    fig.add_trace(go.Scattergl(x=df['timestamp'], y=df[lsc_col], name='LSC',
                               line=dict(color='red', width=1, dash='dash')))
    
    # Limite Inferior (se existir)
    if lic_col in df.columns:
        fig.add_trace(go.Scattergl(x=df['timestamp'], y=df[lic_col], name='LIC',
                                   line=dict(color='red', width=1, dash='dash')))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Tempo",
        yaxis_title="Valor",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- INTERFACE ---

st.title("📊 Medida Móvel Analítica - Performance WebGL")

with st.sidebar:
    st.header("⚙️ Especificações")
    cvi = st.number_input("CVi (%)", value=2.0, step=0.1) / 100
    cvg = st.number_input("CVg (%)", value=5.0, step=0.1) / 100
    cva = st.number_input("Pior Cenário CVa (%)", value=3.0, step=0.1) / 100
    
    st.divider()
    
    # Cálculo das Opções de Meta
    op1 = (0.75 * cvi) * 100
    op2 = (0.5 * cvi) * 100
    op5 = 0.25 * np.sqrt(cvi**2 + cvg**2) * 100
    op6 = 0.375 * np.sqrt(cvi**2 + cvg**2) * 100
    rcv = 2.77 * np.sqrt(cva**2 + cvi**2)

    meta_escolhida = st.selectbox("Meta para Mediana:", [
        f"Opção 1 - EA Desejável ({op1:.2f}%)",
        f"Opção 2 - EA Mínima ({op2:.2f}%)",
        "Opção 3 - Custom (Erro Aleatório)",
        f"Opção 4 - RCV ({rcv:.2f} Absoluto)",
        f"Opção 5 - ES Desejável ({op5:.2f}%)",
        f"Opção 6 - ES Mínima ({op6:.2f}%)",
        "Opção 7 - Custom (Erro Sistemático)"
    ])
    
    valor_manual = 0.0
    if "Opção 3" in meta_escolhida or "Opção 7" in meta_escolhida:
        valor_manual = st.number_input("Valor Customizado (%)", value=10.0) / 100

    n_janela = st.number_input("N (Janela Móvel)", value=10, min_value=2)

file = st.file_uploader("Suba sua planilha (CSV ou Excel)", type=["csv", "xlsx"])

if file:
    if file.name.endswith('.csv'):
        df_input = pd.read_csv(file, sep=None, engine='python')
    else:
        df_input = pd.read_excel(file)
    
    st.info(f"Total de registros: {len(df_input)}")
    
    cols = df_input.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1: col_d = st.selectbox("Coluna Data", cols)
    with c2: col_h = st.selectbox("Coluna Hora", cols)
    with c3: col_r = st.selectbox("Coluna Resultado", cols)

    if st.button("🚀 Processar com Alta Performance"):
        with st.spinner("Executando cálculos robustos..."):
            df_final = processar_dados_robustos(df_input, col_r, col_d, col_h, n_janela)
            
            # Definição de Metas
            mediana_alvo = df_final['res_limpo'].median()
            
            if "Opção 1" in meta_escolhida: perc = op1/100
            elif "Opção 2" in meta_escolhida: perc = op2/100
            elif "Opção 5" in meta_escolhida: perc = op5/100
            elif "Opção 6" in meta_escolhida: perc = op6/100
            elif "Opção 4" in meta_escolhida: perc = rcv 
            else: perc = valor_manual

            if "Opção 4" in meta_escolhida:
                df_final['LSC'] = mediana_alvo + perc
                df_final['LIC'] = mediana_alvo - perc
            else:
                df_final['LSC'] = mediana_alvo * (1 + perc)
                df_final['LIC'] = mediana_alvo * (1 - perc)

            # Limite DP (t-Student 99.999%)
            fator_t = stats.t.ppf(0.99999, n_janela-1)
            df_final['Limite_DP'] = fator_t * (0.75 * cva)

            # --- EXIBIÇÃO DE GRÁFICOS ---
            
            # Mediana Móvel
            fig_mediana = criar_grafico_webgl(df_final, 'Mediana_Movel', 'LSC', 'LIC', 
                                             "Tendência: Mediana Móvel", "Mediana")
            st.plotly_chart(fig_mediana, use_container_width=True)

            # DP Móvel
            fig_dp = criar_grafico_webgl(df_final, 'DP_Movel', 'Limite_DP', 'Limite_DP', 
                                        "Precisão: Desvio Padrão Móvel", "DP")
            # Ajuste para não mostrar LIC no gráfico de DP (que é sempre zero ou positivo)
            fig_dp.data[2].visible = False 
            st.plotly_chart(fig_dp, use_container_width=True)

            # --- ALERTAS ---
            alertas = df_final[(df_final['Mediana_Movel'] > df_final['LSC']) | 
                              (df_final['Mediana_Movel'] < df_final['LIC']) |
                              (df_final['DP_Movel'] > df_final['Limite_DP'])]
            
            if not alertas.empty:
                st.error(f"🚨 {len(alertas)} violações detectadas!")
                st.dataframe(alertas[[col_d, col_h, col_r, 'Mediana_Movel', 'DP_Movel']].head(500))
