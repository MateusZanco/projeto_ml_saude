import argparse
import yaml
import pandas as pd
import joblib
import os
import json
import mlflow
import mlflow.sklearn

# Importa nosso pipeline e todos os modelos/métricas
from data_preprocessing import executar_pipeline_preparacao
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Mapeamento de Nomes para Classes de Modelos (torna o script dinâmico)
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

    # 2. Inicia o experimento no MLflow
    mlflow.set_experiment("Previsao Risco Saude")
    with mlflow.start_run():
        print("Configuração carregada:", config['model_name'])
        mlflow.log_param("model_name", config['model_name'])

        # 3. Executa todo o pipeline de pré-processamento
        (
            X_train, X_test, y_train, y_test,
            scaler, label_encoder, encoded_columns
        ) = executar_pipeline_preparacao(caminho_dados=config['data_path'])

        # 4. Instancia e treina o modelo (com ou sem GridSearchCV)
        base_model = MODELS[config['model_name']](**config.get('static_params', {}))

        if config.get('hyperparameter_grid'):
            print(f"Iniciando GridSearchCV para {config['model_name']}...")
            mlflow.log_param("hyperparameter_tuning", True)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=config['hyperparameter_grid'],
                scoring='f1_weighted', cv=5, verbose=2, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            mlflow.log_params(best_params)
            print("Melhores parâmetros encontrados:", best_params)
        else:
            print(f"Treinando {config['model_name']} com parâmetros base...")
            mlflow.log_param("hyperparameter_tuning", False)
            best_params = config.get('base_params', {})
            best_model = MODELS[config['model_name']](**best_params)
            best_model.fit(X_train, y_train)
            mlflow.log_params(best_params)

        # 5. Avalia o modelo final
        y_pred = best_model.predict(X_test)
        report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        report_str = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print("\n--- Relatório de Classificação no Conjunto de Teste ---")
        print(report_str)

        # 6. LOG NO MLFLOW (O Diário de Bordo do Experimento)
        print("\nLogando experimento no MLflow...")
        weighted_avg = report_dict['weighted avg']
        mlflow.log_metric("f1_score_weighted", weighted_avg['f1-score'])
        mlflow.log_metric("precision_weighted", weighted_avg['precision'])
        mlflow.log_metric("recall_weighted", weighted_avg['recall'])
        mlflow.sklearn.log_model(best_model, "model")
        # Logar pré-processadores como artefatos no MLflow
        joblib.dump(scaler, "scaler.joblib")
        mlflow.log_artifact("scaler.joblib", artifact_path="preprocessors")
        os.remove("scaler.joblib")

        # 7. SALVAR ARTEFATOS PARA PRODUÇÃO
        save_dir = config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        print(f"Salvando artefatos para produção em: {save_dir}")

        joblib.dump(best_model, os.path.join(save_dir, 'best_model.joblib'))
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
        joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.joblib'))
        joblib.dump(encoded_columns, os.path.join(save_dir, 'encoded_columns.joblib'))

        # Salvar as métricas também na pasta do modelo para referência
        results_summary = {
            'best_params': best_params,
            'classification_report': report_dict
        }
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(results_summary, f, indent=4)

        print("\nTreinamento concluído e artefatos salvos com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de treinamento de modelos de ML com MLflow.")
    parser.add_argument('--config', type=str, required=True, help="Caminho para o arquivo de configuração do modelo.")
    args = parser.parse_args()
    train(args.config)
