# Sistema de Previsão de Risco de Saúde

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Streamlit-1.38-red?style=for-the-badge&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-2.14-orange?style=for-the-badge)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](URL_DO_SEU_HUGGING_FACE_SPACE_AQUI)

> [!WARNING]
> **Este conteúdo é destinado apenas para fins educacionais.** Os dados exibidos são ilustrativos e podem não corresponder a situações reais.

---

###  Índice
1. [Visão Geral](#visão-geral-do-projeto)
2. [Demonstração Online](#demonstração-online)
3. [Arquitetura da Solução](#arquitetura-da-solução)
4. [Metodologia e Tecnologias](#metodologia-e-tecnologias)
5. [Como Executar Localmente](#como-executar-localmente)

---

## Visão Geral do Projeto

Este projeto consiste em um pipeline completo de Machine Learning, desenvolvido como atividade final da disciplina de Aprendizado de Máquina. O objetivo é prever o nível de risco de um indivíduo desenvolver uma determinada condição de saúde (classificado em `Baixo`, `Moderado`, `Alto` e `Muito Alto`) com base em um conjunto de dados simulados contendo informações sobre estilo de vida, hábitos e dados clínicos.

A solução abrange desde a análise exploratória e pré-processamento dos dados, passando pelo treinamento e otimização de múltiplos modelos, até o deploy de uma aplicação web interativa.

## Demonstração Online
A aplicação está disponível publicamente no Hugging Face Spaces. Acesse e interaja com o modelo final:

## Arquitetura da Solução
O projeto foi estruturado de forma modular para garantir a organização, manutenibilidade e reprodutibilidade do código.

```bash
projeto_ml_saude/
│
├── artifacts/              # Modelos, scalers e encoders salvos
├── configs/                # Arquivos de configuração (.yaml)
├── data/                   # Conjunto de dados
├── notebooks/              # Notebooks para análise exploratória
├── reports/                # Métricas e figuras geradas pelos treinos
├── src/                    # Código-fonte modularizado
│   ├── data_preprocessing.py # Funções de pré-processamento
│   ├── train.py            # Script genérico para treinamento de modelos
│   ├── predict.py          # Lógica para fazer uma única predição
│   └── app.py              # Código da interface Streamlit
│
├── .gitignore              # Arquivos e pastas a serem ignorados pelo Git
├── requirements.txt        # Lista de dependências do projeto
└── README.md               # Esta documentação
```
## Metodologia e Tecnologias
O projeto foi estruturado seguindo as fases do CRISP-DM, utilizando uma abordagem modular e experimental para garantir a reprodutibilidade e a qualidade do modelo final.

## Como Executar Localmente

Siga os passos abaixo para configurar e executar o projeto em sua máquina.

### Pré-requisitos
- Python 3.13 ou superior
- pip 25.2

### 1. Clonar o Repositório
```bash
git clone [https://github.com/MateusZanco/projeto_ml_saude.git]
```

### 2. **Configurar Ambiente Virtual** 
utilizando o seguinte comando no terminal:  
```bash
python -m venv venv
```

### 3. **Ativar Ambiente Virtual** 
utilizando o seguinte comando no terminal:  
```bash
# Ative o ambiente (Windows)
.\venv\Scripts\activate.ps1
```

```bash
# Em macOS/Linux, use:
 source venv/bin/activate
```

Dica (Windows): Caso ocorra erro de permissão, altere temporiamente a política de execução do PowerShell:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Após ativar, o nome do ambiente virtual aparecerá entre parêntese no início da linha de comando:
```bash
(venv) C:\caminho\para\seuzprojeto
```
### 4. **Instalar as Dependências** 
```bash
pip install -r requirements.txt
```
### 5. **Treinar um Modelo** 
O script src/train.py é genérico e treina o modelo especificado em um arquivo de configuração.
```bash
python src/train.py --config configs/random_forest.yaml
```

### 6. **Visualizar os Experimentos** 
Para ver os resultados salvos pelo MLflow, inicie a interface localmente.  Execute a partir da pasta raiz do projeto
```bash
mlflow ui --workers 1
```

### 7. **Executar a Aplicação Web**
Após o treinamento, o melhor modelo estará salvo na pasta artifacts/. Execute a aplicação Streamlit:
```bash
streamlit run src/app.py
```
A aplicação abrirá automaticamente no seu navegador.

