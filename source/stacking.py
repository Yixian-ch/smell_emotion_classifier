import numpy as np
import pandas as pd
import re
import nltk
import json
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import RegexpTokenizer
# from nltk.wordnet import WordNetLemmatizer
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

nltk.download('punkt', quiet=True)

# Create a directory for reports and models 
def create_report_directory(args):
    """
    Create a directory for storing essential reports and models
    """
    report_dir = Path(args.output)
    
    report_dir.mkdir(exist_ok=True)

    subdirs = ["models", "visualizations"]
    for sub in subdirs:
        sub_dir = report_dir / sub
        sub_dir.mkdir(exist_ok=True)
            
    return report_dir

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="CSV file as training data")
    parser.add_argument("-o", "--output", default="./output_", help="Base folder to save results")
    parser.add_argument("--split_size", type=float, default=0.2, help="Size to split training and testing data")
    return parser.parse_args()

# Preprocessing function with stemming
def preprocessing(text):
    """
    Preprocessing for French text classification with stemming
    """
    # Handle None values
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle spaces around apostrophes like j ' ai -> j'ai
    pattern = re.compile(r"\w+\s*'\s*\w+")
    text = re.sub(pattern, lambda m: m.group().replace(" ", ""), text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Get stopwords list
    with open("./stopwords-fr.txt") as f:
        stop_words = f.read().split()
    
    # Tokenize the text removing punks
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # Initialize French stemmer
    stemmer = FrenchStemmer()
    
    # Apply stemming and remove stopwords
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 1]
    
    # Join tokens back into a string
    return " ".join(tokens)

# Load and process data
def load_data(args):
    """
    Load and preprocess datasets from CSV files
    """
    data = pd.read_csv(args.input)
    # Create numeric labels for emotions
    data['emotion'] = data['emotions'].map({
        'fear': 0,
        'love': 1, 
        'disgust': 2,
        'surprise': 3
    })
    
    # Add word count column
    data['word_count'] = data['excerpt'].str.split().str.len()
    
    # Calculate word count before preprocessing
    emotions_count_before = data.groupby('emotions')['word_count'].sum().to_dict()
    print(f"Before preprocessing each emotion has {emotions_count_before} words")
    
    # Apply preprocessing
    print("Applying preprocessing with stemming")
    data["excerpt_clean"] = data['excerpt'].apply(preprocessing)
    
    # Calculate word count after preprocessing
    data['word_count_clean'] = data['excerpt_clean'].str.split().str.len()
    emotions_count_after = data.groupby('emotions')['word_count_clean'].sum().to_dict()
    print(f"After preprocessing each emotion has {emotions_count_after} words")
    
    return data, emotions_count_before, emotions_count_after

# Generate meta features using cross-validation 
def generate_meta_features(classifiers, X_train_text, y_train, X_test_text, n_classes, cv=5):
    """
    Generate meta-features for stacking using cross-validation
    """
    # Maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Initialize meta-feature matrices
    X_meta_train = np.zeros((len(X_train_text), len(classifiers) * n_classes))
    X_meta_test = np.zeros((len(X_test_text), len(classifiers) * n_classes))
    
    # Store fold performance for each classifier
    cv_results = {}
    
    # For each base classifier
    for i, (name, pipeline) in enumerate(classifiers.items()):
        print(f"Training base classifier: {name}")
        cv_results[name] = []
        
        # Storage for out-of-fold predictions
        train_meta_preds = np.zeros((len(X_train_text), n_classes))
        
        # Storage for test predictions from each fold
        test_meta_preds = np.zeros((len(X_test_text), cv, n_classes))
        
        # K-fold cross-validation
        for j, (train_idx, val_idx) in enumerate(skf.split(X_train_text, y_train)):
            # Split data for current fold
            X_fold_train = [X_train_text[idx] for idx in train_idx]
            X_fold_val = [X_train_text[idx] for idx in val_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            # Train classifier on training fold
            pipeline.fit(X_fold_train, y_fold_train)
            
            # Generate predictions for validation fold
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(X_fold_val)
                train_meta_preds[val_idx] = proba
            else:
                # If classifier doesn't support predict_proba, use one-hot encoding of predictions
                preds = pipeline.predict(X_fold_val)
                for k, pred in enumerate(preds):
                    train_meta_preds[val_idx[k], pred] = 1
            
            # Evaluate on validation fold
            fold_pred = pipeline.predict(X_fold_val)
            fold_accuracy = accuracy_score(y_fold_val, fold_pred)
            fold_f1 = f1_score(y_fold_val, fold_pred, average='macro')
            
            # Store fold results
            cv_results[name].append({
                'fold': j + 1,
                'accuracy': float(fold_accuracy),
                'f1_score': float(fold_f1)
            })
            
            # Generate predictions for test set
            if hasattr(pipeline, "predict_proba"):
                test_meta_preds[:, j] = pipeline.predict_proba(X_test_text)
            else:
                # If classifier doesn't support predict_proba, use one-hot encoding of predictions
                preds = pipeline.predict(X_test_text)
                for k, pred in enumerate(preds):
                    test_meta_preds[k, j, pred] = 1
                
        # Store meta-features - each model contributes n_classes probabilities
        start_col = i * n_classes
        end_col = (i + 1) * n_classes
        X_meta_train[:, start_col:end_col] = train_meta_preds
        
        # Average predictions across folds for test set
        X_meta_test[:, start_col:end_col] = np.mean(test_meta_preds, axis=1)
        
        # Retrain on full training data
        pipeline.fit(X_train_text, y_train)
    
    return X_meta_train, X_meta_test, cv_results


def plot_confusion_matrix(y_test, y_pred, emotion_names, report_dir):
    """
    Create and save confusion matrix visualization
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_names, yticklabels=emotion_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Stacking Model')
    plt.savefig(report_dir / "visualizations" / "confusion_matrix.png")
    plt.close()

def plot_word_distributions(data, report_dir, emotions_count_before, emotions_count_after):
    """
    Create visualizations for word count distributions
    """
    # 1. Histograms of word count distribution for each emotion
    plt.figure(figsize=(15, 10))
    for i, emotion in enumerate(data['emotions'].unique()):
        plt.subplot(2, 2, i+1)
        sns.histplot(data[data['emotions'] == emotion]['word_count'], bins=30)
        plt.title(f'Word Count Distribution - {emotion}')
    plt.tight_layout()
    plt.savefig(report_dir / "visualizations" / "word_count_histograms.png")
    plt.close()

    # 2. Bar plot comparing average word count across emotions
    plt.figure(figsize=(10, 6))
    emotion_word_counts = data.groupby('emotions')['word_count'].mean().sort_values(ascending=False)
    sns.barplot(x=emotion_word_counts.index, y=emotion_word_counts.values)
    plt.title('Average Word Count Comparison Across Emotions')
    plt.savefig(report_dir / "visualizations" / "emotion_word_count_comparison.png")
    plt.close()

    # 3. Comparison of word counts before and after preprocessing
    plt.figure(figsize=(12, 6))
    before_after = pd.DataFrame({
        'emotion': list(emotions_count_before.keys()) + list(emotions_count_after.keys()),
        'word_count': list(emotions_count_before.values()) + list(emotions_count_after.values()),
        'stage': ['Before Preprocessing'] * 4 + ['After Preprocessing'] * 4
    })
    sns.barplot(x='emotion', y='word_count', hue='stage', data=before_after)
    plt.title('Word Count Comparison: Before vs After Preprocessing')
    plt.savefig(report_dir / "visualizations" / "preprocessing_comparison.png")
    plt.close()

def plot_top_words(data, report_dir):
    """
    Create visualization for top 10 words in each emotion category
    """
    # Create a figure with subplots for each emotion
    plt.figure(figsize=(15, 10))
    
    # Get unique emotions
    emotions = data['emotions'].unique()
    
    # For each emotion
    for i, emotion in enumerate(emotions):
        # Get all words for this emotion
        words = ' '.join(data[data['emotions'] == emotion]['excerpt_clean']).split()
        
        # Count word frequencies
        word_freq = pd.Series(words).value_counts()
        
        # Get top 10 words
        top_words = word_freq.head(10)
        
        # Create subplot
        plt.subplot(2, 2, i+1)
        
        # Create horizontal bar plot
        sns.barplot(x=top_words.values, y=top_words.index)
        plt.title(f'Top 10 Words - {emotion}')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
    
    plt.tight_layout()
    plt.savefig(report_dir / "visualizations" / "top_words_by_emotion.png")
    plt.close()
    
    # Also save the raw data to a text file
    with open(report_dir / "top_words_by_emotion.txt", 'w', encoding='utf-8') as f:
        f.write("Top 10 Words for Each Emotion Category:\n\n")
        for emotion in emotions:
            words = ' '.join(data[data['emotions'] == emotion]['excerpt_clean']).split()
            word_freq = pd.Series(words).value_counts()
            top_words = word_freq.head(10)
            
            f.write(f"\n{emotion.upper()}:\n")
            for word, freq in top_words.items():
                f.write(f"{word}: {freq}\n")

# Main function to run the stacking classifier
def run_stacking_classifier():
    """
    Task 4:Main function to run the stacking classifier with preprocessing
    """
    # Parse arguments
    args = get_args()
    
    # Create report directory
    report_dir = create_report_directory(args)
    print(f"Reports will be saved to: {str(report_dir)}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data, emotions_count_before, emotions_count_after = load_data(args)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_word_distributions(data, report_dir, emotions_count_before, emotions_count_after)
    plot_top_words(data, report_dir)
    
    # Check class distribution
    print("\nClass distribution:")
    class_distribution = data['emotions'].value_counts()
    print(class_distribution)
    
    # Map numeric labels to emotion names for better reporting
    emotion_mapping = {0: 'fear', 1: 'love', 2: 'disgust', 3: 'surprise'}
    emotion_names = [emotion_mapping[i] for i in range(len(emotion_mapping))]
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        data["excerpt_clean"], 
        data["emotion"], 
        test_size=args.split_size, 
        random_state=42,
        stratify=data["emotion"]
    )
    
    # Convert Series to lists for easier handling
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    
    # Define base classifiers
    print("\nDefining base classifiers...")
    nb_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
        ('classifier', MultinomialNB(alpha=0.5))
    ])
    
    svm_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ('classifier', SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', random_state=42))
    ])
    
    rf_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ('classifier', RandomForestClassifier(n_estimators=80, max_depth=6, class_weight='balanced', random_state=42))
    ])
    

    base_classifiers = {
        'MultinomialNB': nb_pipeline,
        'SVM': svm_pipeline,
        'RandomForest': rf_pipeline
    }
    

    print("\nGenerating meta-features...")
    X_meta_train, X_meta_test, cv_results = generate_meta_features(
        base_classifiers, 
        X_train_list, 
        y_train, 
        X_test_list,
        len(emotion_mapping),
        cv=5
    )
    

    print("\nTraining meta-classifier...")
    meta_classifier = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    

    meta_classifier.fit(X_meta_train, y_train)
    

    y_meta_pred = meta_classifier.predict(X_meta_test)
    
    # Evaluate stacking model performance
    print("\n===== Stacking Model Performance =====")
    stacking_accuracy = accuracy_score(y_test, y_meta_pred)
    stacking_f1 = f1_score(y_test, y_meta_pred, average='macro')
    stacking_report = classification_report(y_test, y_meta_pred, target_names=emotion_names)
    
    print(f"Accuracy: {stacking_accuracy:.4f}")
    print(f"Macro F1: {stacking_f1:.4f}")
    print("\nClassification Report:")
    print(stacking_report)
    
    # Create and save the confusion matrix
    plot_confusion_matrix(y_test, y_meta_pred, emotion_names, report_dir)
    
    # Create summary report
    summary = {
        "dataset_info": {
            "size": len(data),
            "class_distribution": class_distribution.to_dict()
        },
        "stacking_model": {
            "accuracy": float(stacking_accuracy),
            "f1_score": float(stacking_f1),
            "classification_report": stacking_report,
        },
        "base_classifiers": {}
    }
    
    # Evaluate and add individual base classifiers to summary
    print("\n===== Base Classifier Performance =====")
    
    for name, pipeline in base_classifiers.items():
        # Get predictions
        y_pred = pipeline.predict(X_test_list)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1:.4f}")
        
        # Add to summary
        summary["base_classifiers"][name] = {
            "accuracy": float(accuracy),
            "f1_score": float(f1)
        }
    
    # Save the combined summary report
    with open(report_dir / "results_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save only the final stacking model (most important)
    model_info = {
        "meta_classifier": meta_classifier,
        "base_classifiers": base_classifiers,
        "emotion_mapping": emotion_mapping
    }
    
    with open(report_dir / "models" / "stacking_model.pkl", 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\nResults and model saved to {str(report_dir)}")


if __name__ == "__main__":
    run_stacking_classifier()
