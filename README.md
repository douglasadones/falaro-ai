# Projeto de Detecção de Desinformação (Fake News) em Português
## Visão Geral
Este projeto implementa um classificador de Machine Learning para identificar notícias falsas em português, utilizando o algoritmo Passive Aggressive Classifier (PAC) e a vetorização TF-IDF.
## Estrutura do Projeto
- `main.py`: Script principal que carrega o dataset, pré-processa os dados, treina o modelo, avalia o desempenho e gera a matriz de confusão.
- `Fake.br-Corpus/`: Diretório contendo o dataset de notícias em português.
- `matriz_confusao.png`: Visualização da avaliação do modelo.
- `relatorio_classificacao.txt`: Métricas de desempenho do modelo.
## Como Executar
1. **Instalar dependências:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn matplotlib seaborn nltk
python -c "import nltk; nltk.download('stopwords')"
```
2. **Executar o script principal:**
```bash
python main.py
```