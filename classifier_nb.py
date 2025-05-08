import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df_surprise = pd.read_csv("surprise.csv")
df_fear = pd.read_csv("fear.csv")
df_love = pd.read_csv("love.csv")
df_disgust = pd.read_csv("disgust.csv")
df = pd.concat([df_surprise, df_fear, df_disgust, df_love])

X = df['text']
y = df['emotion']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=25, stratify=y
)

vectorizer = TfidfVectorizer(stop_words=list(fr_stop))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

nb = MultinomialNB()
trained_model = nb.fit(X_train, y_train)
predictions = trained_model.predict(X_test)
report = classification_report(y_test, predictions)
print("Naive Bayes Classification Report:\n", report)

feature_names = vectorizer.get_feature_names_out()
class_labels = nb.classes_
log_probs = nb.feature_log_prob_
top_n = 20

for i, label in enumerate(class_labels):
    print(f"\nTop {top_n} words for class '{label}':")
    top_features = log_probs[i].argsort()[::-1][:top_n]
    for idx in top_features:
        word = feature_names[idx]
        weight = log_probs[i][idx]
        print(f"{word}\t{weight:.3f}")