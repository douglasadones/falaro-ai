import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re
import os

print("1. Carregando e preparando o dataset Fake.br Corpus...")

def load_fakebr_corpus(base_path="Fake.br-Corpus/full_texts"):
    data = []
    for category in ["fake", "true"]:
        category_path = os.path.join(base_path, category)
        
        if not os.path.isdir(category_path):
            print(f"Aviso: Diretório não encontrado: {category_path}")
            continue
            
        for news_file in os.listdir(category_path):
            if news_file.endswith(".txt"):
                news_path = os.path.join(category_path, news_file)
                try:
                    with open(news_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        label = 1 if category == "fake" else 0
                        data.append({"text": text, "label": label})
                except Exception as e:
                    pass
    
    if not data:
        print("ERRO CRÍTICO: Nenhuma notícia foi carregada. Verifique o caminho do dataset.")
        return pd.DataFrame(columns=['text', 'label'])

    df = pd.DataFrame(data)
    df = df[df['text'].str.strip().astype(bool)]
    print(f"Dataset carregado. Total de notícias: {len(df)}")
    print(f"Notícias Falsas (1): {df['label'].sum()}")
    print(f"Notícias Verdadeiras (0): {len(df) - df['label'].sum()}")
    return df

df = load_fakebr_corpus()

print("2. Pré-processamento de texto...")
try:
    stop_words = set(stopwords.words('portuguese'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-záàâãéèêíìîóòôõúùûç\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(preprocess_text)

X = df['text']
y = df['label']
if len(y.unique()) < 2:
    print("Erro: O dataset contém apenas uma classe. Não é possível treinar um classificador binário.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("4. Extração de Features (TF-IDF)...")
tfidf_vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

print("5. Treinamento do Modelo (Passive Aggressive Classifier)...")
pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
pac.fit(tfidf_train, y_train)

print("6. Avaliação do Modelo...")
y_pred = pac.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Verdadeira (0)', 'Falsa (1)'])

print(f"\nAcurácia: {accuracy:.4f}")
print("\nMatriz de Confusão:\n", conf_mat)
print("\nRelatório de Classificação:\n", class_report)

print("7. Gerando visualização da Matriz de Confusão...")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predito: Verdadeira', 'Predito: Falsa'], yticklabels=['Real: Verdadeira', 'Real: Falsa'])
plt.title('Matriz de Confusão do Passive Aggressive Classifier')
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Predito')
plt.savefig('matriz_confusao.png')
plt.close()
print("Visualização salva em 'matriz_confusao.png'")

def prever_noticia(noticia):
    noticia_processada = preprocess_text(noticia)
    noticia_vetorizada = tfidf_vectorizer.transform([noticia_processada])
    predicao = pac.predict(noticia_vetorizada)[0]
    
    resultado = "Falsa" if predicao == 1 else "Verdadeira"
    return resultado

print("\n8. Demonstração de Previsão:")
exemplo_fake = "Vacina causa autismo, diz estudo polêmico e sem provas científicas."
exemplo_true = "Cientistas da USP descobrem nova espécie de peixe no Oceano Atlântico, conforme artigo publicado na Nature."

print(f"Notícia: '{exemplo_fake}' -> Classificação: {prever_noticia(exemplo_fake)}")
print(f"Notícia: '{exemplo_true}' -> Classificação: {prever_noticia(exemplo_true)}")

with open('relatorio_classificacao.txt', 'w') as f:
    f.write(f"Acurácia: {accuracy:.4f}\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(conf_mat) + "\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(class_report)

print("Relatório de classificação salvo em 'relatorio_classificacao.txt'")