import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --- Funções Auxiliares (suas funções originais, já estavam ótimas) ---

def remover_outliers_iqr(df, nome_coluna):
    """Remove outliers de uma coluna específica de um DataFrame usando o método IQR."""
    Q1 = df[nome_coluna].quantile(0.25)
    Q3 = df[nome_coluna].quantile(0.75)
    Amp_interquartil = Q3 - Q1
    limite_inferior = Q1 - (1.5 * Amp_interquartil)
    limite_superior = Q3 + (1.5 * Amp_interquartil)
    df_filtrado = df[df[nome_coluna].between(limite_inferior, limite_superior)]
    return df_filtrado

def substituir_zeros_pela_mediana(df, nome_coluna):
    """Substitui valores 0 pela mediana da coluna (calculada sem os zeros)."""
    mediana = df[df[nome_coluna] > 0][nome_coluna].median()
    df.loc[df[nome_coluna] == 0, nome_coluna] = mediana
    return df

# --- Funções Principais Refatoradas ---

def limpar_dados_numericos(df):
    """
    Aplica a remoção de outliers em todas as colunas numéricas e
    corrige os zeros na coluna de água.
    """
    print("Iniciando limpeza de dados numéricos...")
    df_processado = df.copy()

    df_processado["Calorias"] = pd.to_numeric(df_processado["Calorias"], errors="coerce")
    df_processado["Colesterol"] = pd.to_numeric(df_processado["Colesterol"], errors="coerce")
    df_processado["Passos_Diarios"] = pd.to_numeric(df_processado["Passos_Diarios"], errors="coerce")

    df_processado = df_processado.dropna()

    df_processado = df_processado.drop(columns=["ID"], errors="ignore")

    colunas_quantitativas = df_processado.select_dtypes(include=np.number).columns.tolist()
    
    for coluna in colunas_quantitativas:
        tamanho_antes = len(df_processado)
        df_processado = remover_outliers_iqr(df_processado, coluna)
        tamanho_depois = len(df_processado)
        print(f"  - Coluna '{coluna}': {tamanho_antes - tamanho_depois} outliers removidos.")

    df_processado = substituir_zeros_pela_mediana(df_processado, 'Agua_Litros')
    print("Limpeza de dados numéricos concluída.")
    return df_processado

def codificar_features_e_alvo(X, y):
    """
    Aplica One-Hot Encoding em X e Label Encoding em y.
    Retorna os dados, o encoder do alvo e as colunas do X codificado.
    """
    colunas_categoricas = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True, dtype=int)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # MELHORIA: Retornar também as colunas codificadas
    encoded_columns = X_encoded.columns.tolist()
    
    return X_encoded, y_encoded, label_encoder, encoded_columns

def dividir_balancear_normalizar(X_encoded, y_encoded):
    """
    Divide os dados, aplica SMOTE no treino e normaliza ambos os conjuntos.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Dados divididos. Treino: {len(X_train)} amostras, Teste: {len(X_test)} amostras.")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("SMOTE aplicado no conjunto de treino.")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    print("Dados de treino e teste normalizados.")

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler


def executar_pipeline_preparacao(caminho_dados):
    """
    Orquestra todo o pipeline de pré-processamento de dados.
    
    Args:
        caminho_dados (str): Caminho para o arquivo CSV.
        
    Returns:
        Múltiplos objetos: Todos os dataframes e artefatos necessários para o treino.
    """
    # 1. Carregar os dados
    df = pd.read_csv(caminho_dados)
    
    # 2. Limpar dados numéricos (outliers e zeros)
    df_limpo = limpar_dados_numericos(df)
    
    # 3. Separar features (X) e alvo (y)
    X = df_limpo.drop('Risco_Doenca', axis=1)
    y = df_limpo['Risco_Doenca']
    
    # 4. Codificar features e alvo
    X_encoded, y_encoded, label_encoder, encoded_columns = codificar_features_e_alvo(X, y)
    
    # 5. Dividir, balancear e normalizar
    X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler = dividir_balancear_normalizar(X_encoded, y_encoded)
    
    # Retorna tudo o que é necessário para treinar e para futuras predições
    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler, label_encoder, encoded_columns