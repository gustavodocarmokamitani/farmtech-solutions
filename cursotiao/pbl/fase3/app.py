import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# 1. Configuração da Página do Streamlit
st.set_page_config(
    page_title="FarmTech Solutions - Dashboard",
    page_icon="🌱",
    layout="wide"
)

# 2. Carregamento dos Dados e Treinamento do Modelo Preditivo de Suporte
@st.cache_data
def carregar_dados():
    df = pd.read_csv('Atividade_Cap10_produtos_agricolas.csv')
    return df

@st.cache_resource
def treinar_modelo(df):
    X = df.drop('label', axis=1)
    y = df['label']
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X, y)
    return modelo

df = carregar_dados()
modelo = treinar_modelo(df)

# 3. Título Principal
st.title("🌱 FarmTech Solutions — Monitoramento Agrícola Inteligente")
st.markdown("Bem-vindo ao painel de controle operacional e de recomendação preditiva.")
st.divider()

# 4. Barra Lateral (Inputs do Usuário / Sensores)
st.sidebar.header("📡 Parâmetros dos Sensores em Tempo Real")

# Inputs de Solo
st.sidebar.subheader("Solo")
input_n = st.sidebar.slider("Nitrogênio (N)", int(df['N'].min()), int(df['N'].max()), int(df['N'].mean()))
input_p = st.sidebar.slider("Fósforo (P)", int(df['P'].min()), int(df['P'].max()), int(df['P'].mean()))
input_k = st.sidebar.slider("Potássio (K)", int(df['K'].min()), int(df['K'].max()), int(df['K'].mean()))
input_ph = st.sidebar.slider("pH do Solo", float(df['ph'].min()), float(df['ph'].max()), float(df['ph'].mean()), step=0.1)

# Inputs de Clima
st.sidebar.subheader("Clima")
input_temp = st.sidebar.slider("Temperatura (°C)", float(df['temperature'].min()), float(df['temperature'].max()), float(df['temperature'].mean()), step=0.5)
input_hum = st.sidebar.slider("Umidade do Ar (%)", float(df['humidity'].min()), float(df['humidity'].max()), float(df['humidity'].mean()), step=1.0)
input_rain = st.sidebar.slider("Precipitação/Chuva (mm)", float(df['rainfall'].min()), float(df['rainfall'].max()), float(df['rainfall'].mean()), step=5.0)

# 5. Lógica de Negócio (Irrigação e Predição)
status_irrigacao = "⚠️ Desligada (Solo Seco)" if input_hum < 50 else "✅ Ativa (Mantendo Umidade)"
if input_rain > 150:
    sugestao_irrigacao = "🚫 Suspender Irrigação (Previsão de Chuva Forte detectada)"
elif input_hum < 40 and input_temp > 28:
    sugestao_irrigacao = "💧 Aumentar Vazão (Clima Crítico: Quente e Seco)"
else:
    sugestao_irrigacao = "🔄 Irrigação Inteligente Automatizada (Fluxo Padrão)"

# Predição da Melhor Cultura usando o modelo treinado
dados_sensor = np.array([[input_n, input_p, input_k, input_temp, input_hum, input_ph, input_rain]])
cultura_recomendada = modelo.predict(dados_sensor)[0]

# 6. Layout do Painel Principal
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Status Atual do Sistema de Irrigação", value=status_irrigacao)

with col2:
    st.metric(label="Recomendação de IA para Plantio", value=cultura_recomendada.upper())

with col3:
    st.info(f"**Sugestão de Irrigação:** \n{sugestao_irrigacao}")

st.divider()

# 7. Visualizações Gráficas Interativas
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("📊 Distribuição Atual dos Nutrientes e pH")
    df_atual = pd.DataFrame({
        'Atributo': ['Nitrogênio (N)', 'Fósforo (P)', 'Potássio (K)', 'pH do Solo'],
        'Valor Atual': [input_n, input_p, input_k, input_ph]
    })
    fig_bar = px.bar(df_atual, x='Atributo', y='Valor Atual', color='Atributo', text_auto=True,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_bar, width='stretch')

with col_graf2:
    st.subheader("🌦️ Clima: Temperatura vs Umidade na Base Histórica")
    fig_scatter = px.scatter(df, x='temperature', y='humidity', color='label', 
                             title="Histórico de Culturas vs Condição Selecionada (Ponto Vermelho)")
    fig_scatter.add_scatter(x=[input_temp], y=[input_hum], name="Sensor Atual", 
                            marker=dict(color='red', size=15, symbol='triangle-up'))
    st.plotly_chart(fig_scatter, width='stretch')

# 8. Tabela de Dados Históricos
if st.checkbox("Mostrar Base de Dados Histórica Completa"):
    st.subheader("📋 Dados de Culturas da FarmTech Solutions")
    st.dataframe(df)
