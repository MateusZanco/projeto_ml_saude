
import argparse
import yaml
import pandas as pd
import joblib
import os
import json

from data_preprocessing import executar_pipeline_preparacao
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --- Mapeamento de Nomes para Classes de Modelos ---
MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "XGBClassifier": XGBClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC
}

def train(config_path):
    # 1. Carrega a configuração do arquivo .yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("Configuração carregada:", config['model_name'])
    
    # 2. Executa todo o pipeline de pré-processamento com uma única função
    (
        X_train, X_test, y_train, y_test, 
        scaler, label_encoder, encoded_columns
    ) = executar_pipeline_preparacao(caminho_dados=config['data_path'])
    
    # 3. Instancia o modelo dinamicamente a partir da configuração
    base_model = MODELS[config['model_name']](**config.get('static_params', {}))
    
    # 4. Decide se vai fazer a busca de hiperparâmetros ou treino rápido
    if config.get('hyperparameter_grid'):
        print(f"Iniciando GridSearchCV para {config['model_name']}...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=config['hyperparameter_grid'],
            scoring='f1_weighted', cv=5, verbose=2, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print("Melhores parâmetros encontrados:", best_params)
    else:
        print(f"Treinando {config['model_name']} com parâmetros base...")
        best_params = config['base_params']
        best_model = MODELS[config['model_name']](**best_params)
        best_model.fit(X_train, y_train)

    # 5. Avaliação do modelo no conjunto de teste
    y_pred = best_model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    print("\n--- Relatório de Classificação no Conjunto de Teste ---")
    print(report_str)

    # 6. Salvar resultados e artefatos
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Salvando artefatos em: {save_dir}")
    joblib.dump(best_model, os.path.join(save_dir, 'best_model.joblib'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.joblib'))
    joblib.dump(encoded_columns, os.path.join(save_dir, 'encoded_columns.joblib'))
    
    results = {
        'best_params': best_params,
        'classification_report': report_dict
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nTreinamento concluído e artefatos salvos com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de treinamento de modelos de ML.")
    parser.add_argument('--config', type=str, required=True, help="Caminho para o arquivo de configuração do modelo (ex: configs/random_forest.yaml).")
    args = parser.parse_args()
    
    train(args.config)