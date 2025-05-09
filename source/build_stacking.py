import numpy as np
import pandas as pd
import re
import nltk
import json
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from argparse import ArgumentParser
from sklearn.ensemble import StackingClassifier
from stacking import load_data, preprocessing, get_args


args = get_args()

data,s,t = load_data(args)

X,y = data['excerpt_clean'], data['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ("MultinomialNB",Pipeline(
        [
        ('vectorizer', CountVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
        ('classifier', MultinomialNB(alpha=0.5))        
    ])),
    ("SVM",Pipeline(
        [("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)),
         ('classifier', SVC(kernel='linear',C=1.0, probability=True, class_weight='balanced', random_state=42))
         ]
    )),
    ("RF",Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ('classifier', RandomForestClassifier(n_estimators=80, max_depth=6, class_weight='balanced',random_state=42))
    ]))
]


stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

