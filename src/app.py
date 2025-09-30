import streamlit as st
import pandas as pd
from predict import make_prediction 


st.set_page_config(
    page_title="Previsão de Risco de Saúde",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 450px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Sistema de Previsão de Risco de Saúde")

st.warning(
    """
    **Atenção:** Este conteúdo é destinado apenas para fins educacionais. 
    Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
    """,
    icon="⚠️"
)

st.markdown("Insira os dados do paciente na barra lateral à esquerda para obter uma previsão do risco de doença.")


with st.sidebar:
    st.header("Dados do Paciente")

    with st.expander("Informações Pessoais e Hábitos", expanded=True):
        idade = st.number_input("Idade", min_value=0, max_value=120, value=45, step=1)
        sexo = st.radio("Sexo", ["Masculino", "Feminino"], index=0)
        altura_cm = st.number_input("Altura (cm)", min_value=0, value=175)
        peso_kg = st.number_input("Peso (kg)", min_value=0.0, value=87.0, format="%.1f")
        fumante = st.radio("É Fumante?", ["Sim", "Não"], index=1)
        alcool = st.selectbox("Consumo de Álcool", ["Não consome", "Baixo", "Moderado", "Alto"], index=1)
        historico = st.radio("Histórico Familiar?", ["Sim", "Não"], index=0)

    with st.expander("🏃 Estilo de Vida e Rotina"):
        passos = st.number_input("Média de Passos Diários", min_value=0, value=8000)
        sono = st.slider("Média de Horas de Sono", 4.0, 12.0, 7.0, 0.5)
        agua = st.slider("Média de Litros de Água por Dia", 1.0, 5.0, 2.5, 0.5)
        calorias = st.number_input("Média de Calorias Ingeridas", min_value=0, value=2200)
        trabalho = st.number_input("Média de Horas de Trabalho por Dia", min_value=0, value=8)

    with st.expander("Dados Clínicos"):
        fc_repouso = st.number_input("Frequência Cardíaca em Repouso", min_value=0, value=70)
        colesterol = st.number_input("Nível de Colesterol (mg/dL)", min_value=0, value=210)
        pa_sistolica = st.number_input("Pressão Arterial Sistólica", min_value=0, value=130)
        pa_diastolica = st.number_input("Pressão Arterial Diastólica", min_value=0, value=85)


if st.sidebar.button("Analisar Risco", use_container_width=True, type="primary"):
    

    if altura_cm > 0:
        imc = peso_kg / ((altura_cm / 100) ** 2)
    else:
        imc = 0

    input_data = {
        'Idade': idade, 'Sexo': sexo, 'IMC': imc, 'Passos_Diarios': passos,
        'Horas_Sono': sono, 'Agua_Litros': agua, 'Calorias': calorias,
        'Fumante': fumante, 'Alcool': alcool, 'Horas_Trabalho': trabalho,
        'Frequencia_Cardiaca_Repouso': fc_repouso, 'Pressao_Sistolica': pa_sistolica,
        'Pressao_Diastolica': pa_diastolica, 'Colesterol': colesterol, 'Historico_Familiar': historico
    }
    

    prediction, probabilities = make_prediction(input_data)
    

    st.header("Resultado da Análise", divider='rainbow')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="IMC Calculado", value=f"{imc:.2f}")
    with col2:
        st.metric(label="Nível de Risco Previsto", value=prediction)
        
    st.subheader("Confiança por Classe")
    
    df_probs = pd.DataFrame(list(probabilities.items()), columns=['Classe', 'Probabilidade'])
    df_probs = df_probs.sort_values(by='Probabilidade', ascending=False)
    
    st.dataframe(
        df_probs,
        column_config={
            "Classe": "Classe de Risco",
            "Probabilidade": st.column_config.ProgressColumn(
                "Probabilidade",
                format="%.2f%%",
                min_value=0,
                max_value=1,
            ),
        },
        hide_index=True,
        use_container_width=True
    )

