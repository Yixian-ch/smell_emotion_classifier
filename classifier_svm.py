import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

nlp = spacy.load('fr_core_news_sm')
fr_stop = nlp.Defaults.stop_words

df = pd.read_csv("corpus.csv")
X = df['text']
y = df['emotion']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25, stratify=y
)

leaky_disgust = [
    'dégoût', 'désagréable', 'nauséeuse', 'nausées', 'puanteur',
    'puant', 'nauséabonde', 'répugnance', 'infecte', 'putréfaction',
    'odorant', 'odorante', 'odoriférant', 'odoriférante', 'odeur', 'repoussante', 'amère', 'détestable','mauvais'
    ,'amer', 'insupportable', 'saveur', 'malade', 'pourri', 'fétide' 'répugnante', 'âcre'
]


leaky_fear = [
    'danger', 'dangereux', 'dangereuse', 'dangereuses',
    'craindre', 'craint', 'affreux', 'redoutable', 'terrible', 
    'terribles', 'violente', 'violent', 'violents', 'empoisonner', 
    'poison', 'poisons', 'tuer', 'mort'
]


amour = [c, 'amour', 'amou','coeur', 'aime', 'aimée', 'amant', 'cœur', 'amants', 'aimer', 'affections', 'amoureuse', 'charmes']
surprise = []
fear = ["peur"]

leaky_love = [
    'amour', 'aimé', 'aimait', 'amande', 'amandes', 'amoureuse',
    'amoureusement', 'amante', 'tendre', 'tendresse', 'tendres', 
    'femme', 'baiser', 'baisers', 'affection', 'passion', 'sentiments','amoureux', 'amours', 'amour', 'amou', 'amans','coeur', 'aime', 'aimée', 'amant', 'cœur', 'amants', 'aimer', 'affections', 'amoureuse', 'parfumée', 'parfumé', 'odorantes',
    'amoureuses', 'parfumés', 'parfum', 'beauté', 'âme', 'rose', 'roses', 'cœurs', 'douce', 'aimées', 'coeurs'
]


leaky_surprise = [
    'étonné', 'étonnement', 'étonne', 'étonnant', 'étonnante', 
    'étonner', 'étonnait', 'étonnés', 'surprend', 'surprenant', 
    'étrange', 'étranges', 'choc', 'brusquement', 'oh', "surpris", "surprise"
]


other = ["fleurs", 'parfums', "eft", 'qu', 'parfumé', 'parfumée', 'parfume', 'sc', 'beaucoup', 'quelqu','lorsqu', 'parfumées']

french_stp = list(fr_stop) + leaky_disgust + leaky_love + leaky_fear + leaky_surprise + other

vectorizer = TfidfVectorizer(stop_words=french_stp, min_df=2)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

svm = LinearSVC()
trained_model = svm.fit(X_train, y_train)
predictions = trained_model.predict(X_test)

print("SVM Classification Report:\n", classification_report(y_test, predictions))

disp = ConfusionMatrixDisplay.from_estimator(
    trained_model,
    X_test,
    y_test,
    display_labels=trained_model.classes_,
    cmap=plt.cm.Blues
)
plt.title("Confusion Matrix - Linear SVM")
plt.show()

feature_names = vectorizer.get_feature_names_out()
class_labels = trained_model.classes_
top_n = 20

top_words_dict = {}

for i, class_label in enumerate(class_labels):
    top_features = np.argsort(trained_model.coef_[i])[-top_n:][::-1]
    top_words = [feature_names[j] for j in top_features]
    print(f"Top {top_n} words for class '{class_label}':\n{top_words}\n")
    top_words_dict[class_label] = top_words

