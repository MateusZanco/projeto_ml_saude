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

st.markdown(
    """
    O objetivo desta aplicação é prever o nível de risco de um indivíduo desenvolver uma doença, 
    classificado em **Baixo, Moderado, Alto e Muito Alto**. A previsão é baseada em um conjunto de dados simulados 
    contendo informações sobre estilo de vida, hábitos e dados clínicos, utilizando um modelo de Machine Learning.
    """
)

st.warning(
    """
    **Atenção:** Este conteúdo é destinado apenas para fins educacionais. 
    Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
    """,
    icon="⚠️"
)

st.markdown("Insira os seus dados na barra lateral à esquerda para obter uma previsão do risco de doença.")


with st.sidebar:
    st.header("Dados do Paciente")

    with st.expander("Informações Pessoais e Hábitos", expanded=True):
        idade = st.number_input("Idade", min_value=18, max_value=89, value=45, step=1, help="Insira a sua idade em anos completos.")
        sexo = st.radio("Sexo", ["Masculino", "Feminino"], index=0)
        altura_cm = st.number_input("Altura (cm)", min_value=145, max_value=200, value=175, help="Insira a sua altura em centímetros (ex: 175).")
        peso_kg = st.number_input("Peso (kg)", min_value=40.0, max_value=140.0, value=87.0, format="%.1f", help="Insira o seu peso em quilogramas (ex: 87.5).")
        fumante = st.radio("É Fumante?", ["Sim", "Não"], index=1, help="Marque 'Sim' se você fuma regularmente.")
        alcool = st.selectbox("Consumo de Álcool", ["Não consome", "Baixo", "Moderado", "Alto"], index=1, help="Indique a frequência de consumo de bebidas alcoólicas.")
        historico = st.radio("Histórico Familiar?", ["Sim", "Não"], index=0, help="Marque 'Sim' se há caso de doença na família próxima (pais, irmãos).")

    with st.expander("Estilo de Vida e Rotina"):
        passos = st.number_input("Média de Passos Diários", min_value=1000, max_value=20000, value=8000, help="Estimativa média de passos por dia.")
        sono = st.slider("Média de Horas de Sono", 4.0, 12.0, 7.0, 0.5, help="Média de horas de sono.")
        agua = st.slider("Média de Litros de Água por Dia", 1.0, 5.0, 2.5, 0.5, help="Média de litros de água consumidos por dia.")
        calorias = st.number_input("Média de Calorias Ingeridas", min_value=1200, max_value=7000, value=2200, help="Estimativa da média de calorias ingeridas diariamente.")
        trabalho = st.number_input("Média de Horas de Trabalho por Dia", min_value=4, max_value=13, value=8, help="Média de horas dedicadas ao trabalho por dia.")

    with st.expander("Dados Clínicos"):
        fc_repouso = st.number_input("Frequência Cardíaca em Repouso", min_value=50, max_value=100, value=70, help="Batimentos por minuto (BPM) medidos em repouso.")
        colesterol = st.number_input("Nível de Colesterol (mg/dL)", min_value=120, max_value=380, value=210, help="Nível total de colesterol em miligramas por decilitro (mg/dL).")
        pa_sistolica = st.number_input("Pressão Arterial Sistólica", min_value=90, max_value=180, value=130, help="O valor 'máximo' da medição da pressão arterial (ex: 120 em 120/80).")
        pa_diastolica = st.number_input("Pressão Arterial Diastólica", min_value=60, max_value=120, value=85, help="O valor 'mínimo' da medição da pressão arterial (ex: 80 em 120/80).")


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

