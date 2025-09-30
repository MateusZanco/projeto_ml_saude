import joblib
import pandas as pd
import os

# --- Carregar os Artefatos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, '../artifacts/random_forest/')

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.joblib')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib')
ENCODED_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, 'encoded_columns.joblib')

# Carrega os objetos salvos
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
encoded_columns = joblib.load(ENCODED_COLUMNS_PATH)

def make_prediction(input_data):
    """
    Recebe os dados de um novo paciente, processa e retorna a predição.
    
    Args:
        input_data (dict): Dicionário com os dados do paciente.
        
    Returns:
        tuple: Uma tupla contendo (str: predição decodificada, dict: probabilidades por classe).
    """
    # 1. Converter o dicionário de entrada para um DataFrame do pandas
    df = pd.DataFrame([input_data])

    # 2. Aplicar One-Hot Encoding
    # Isso irá criar colunas para as categorias presentes nos dados de entrada
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    # 3. Alinhar as colunas com as do modelo treinado
    # Garante que o dataframe tenha exatamente as mesmas colunas que o modelo espera
    df_aligned = df_encoded.reindex(columns=encoded_columns, fill_value=0)

    # 4. Normalizar os dados usando o scaler JÁ TREINADO
    df_scaled = scaler.transform(df_aligned)

    # 5. Fazer a predição de classe e de probabilidades
    prediction_encoded = model.predict(df_scaled)
    prediction_proba = model.predict_proba(df_scaled)

    # 6. Decodificar a predição para o nome da classe original (ex: 'Alto')
    prediction_decoded = label_encoder.inverse_transform(prediction_encoded)[0]

    # 7. Formatar as probabilidades em um dicionário legível
    probabilities = {label_encoder.classes_[i]: f"{proba:.2%}" for i, proba in enumerate(prediction_proba[0])}

    return prediction_decoded, probabilities
