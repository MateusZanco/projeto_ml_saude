import gradio as gr
from predict import make_prediction

def predict_risk(idade, sexo, imc, passos, sono, agua, calorias, fumante, alcool, trabalho, fc_repouso, pa_sistolica, pa_diastolica, colesterol, historico):
    
    input_data = {
        'Idade': idade, 'Sexo': sexo, 'IMC': imc, 'Passos_Diarios': passos,
        'Horas_Sono': sono, 'Agua_Litros': agua, 'Calorias': calorias,
        'Fumante': fumante, 'Alcool': alcool, 'Horas_Trabalho': trabalho,
        'Frequencia_Cardiaca_Repouso': fc_repouso, 'Pressao_Sistolica': pa_sistolica,
        'Pressao_Diastolica': pa_diastolica, 'Colesterol': colesterol, 'Historico_Familiar': historico
    }
    
    prediction, probabilities = make_prediction(input_data)
    

    output_probabilities = f"Probabilidades:\n" + "\n".join([f"- {classe}: {prob}" for classe, prob in probabilities.items()])
    
    return prediction, output_probabilities


DISCLAIMER = """
<p style='text-align: center; font-size: 12px;'>
Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
</p>
"""

# Lista de componentes de entrada
inputs = [
    gr.Number(label="Idade", value=45),
    gr.Radio(label="Sexo", choices=["Masculino", "Feminino"], value="Masculino"),
    gr.Slider(label="IMC (Índice de Massa Corporal)", minimum=15, maximum=50, value=28),
    gr.Number(label="Média de Passos Diários", value=8000),
    gr.Slider(label="Média de Horas de Sono", minimum=4, maximum=12, step=0.5, value=7),
    gr.Slider(label="Média de Litros de Água por Dia", minimum=1, maximum=5, step=0.5, value=2.5),
    gr.Number(label="Média de Calorias Ingeridas", value=2200),
    gr.Radio(label="É Fumante?", choices=["Sim", "Não"], value="Não"),
    gr.Radio(label="Consumo de Álcool", choices=["Baixo", "Moderado", "Alto", "Não consome"], value="Não consome"),
    gr.Number(label="Média de Horas de Trabalho por Dia", value=8),
    gr.Number(label="Frequência Cardíaca em Repouso", value=70),
    gr.Number(label="Pressão Arterial Sistólica", value=120),
    gr.Number(label="Pressão Arterial Diastólica", value=80),
    gr.Number(label="Nível de Colesterol", value=200),
    gr.Radio(label="Possui Histórico Familiar da Doença?", choices=["Sim", "Não"], value="Não"),
]

outputs = [
    gr.Label(label="Risco Previsto"),
    gr.Textbox(label="Probabilidades por Classe", lines=5)
]

iface = gr.Interface(
    fn=predict_risk,
    inputs=inputs,
    outputs=outputs,
    title="🩺 Sistema de Previsão de Risco de Saúde",
    description="Insira os dados do paciente para obter uma previsão do risco de doença com base no modelo de Machine Learning treinado com Random Forest.",
    article=DISCLAIMER,
    theme=gr.themes.Soft()
)

iface.launch()
