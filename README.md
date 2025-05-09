# Feelings Finder: Classifying Emotions from Olfactory Contexts

This repository contains the code and documentation for our emotion classification project, which identifies emotions (Love, Disgust, Fear, and Surprise) from textual descriptions of olfactory experiences extracted from historical French texts.

## Project Overview

We developed a stacking ensemble model that combines multiple classical machine learning approaches (SVM, Random Forest, and Multinomial Naive Bayes) with a logistic regression meta-learner to classify emotions from olfactory contexts.


## Data

Our dataset comes from the [Odeuropa Smell Explorer project](https://explorer.odeuropa.eu/smells), which contains historical texts with olfactory contexts. We crawled and processed 2,593 documents across four emotion categories:
- Love (798 documents)
- Disgust (801 documents)
- Fear (490 documents)
- Surprise (504 documents)

You can access our processed data on [Google Drive](https://drive.google.com/drive/folders/1KXq1Ulc01vNQN3bL8O-58WdWGYS39En4).

## Scripts Usage

### Data Collection

1. Web Crawling:

python webcrawl.py -bu <base_url> -od <output_directory> -pr <start_page> <end_page(inclusive)>
```bash
python webcrawl.py -bu "https://explorer.odeuropa.eu/api/search?filter_emotion=http://data.odeuropa.eu/vocabulary/plutchik/disgust&filter_language=fr&hl=en&page={page}&sort=&type=smells" -pr 1 5 -od "data/disgust_output"
```
   - Output: [data](https://drive.google.com/drive/folders/1KXq1Ulc01vNQN3bL8O-58WdWGYS39En4)

   **Organization:**
   - `raw_data/`
     - Contains the raw web-scraped documents
     - Preserved in case we need to extract more metadata later

   - `data/`
     - Filtered from `raw_data/` to include:

       - Metadata relevant to our task
       - The actual excerpts (texts)


2. Filtering and Formatting the data (multistep process):

```bash
# Step 1: Filter the corpus to keep only articles with the target emotion
# Repeat this for each emotion (fear, love, disgust, surprise)
python filter_data-emotion.py -ip disgust_output/data -op task_1_output/disgust -e disgust -t filter_corpus

python filter_data-emotion.py -ip love_output/data -op task_1_output/love -e love -t filter_corpus

python filter_data-emotion.py -ip fear_output/data -op task_1_output/fear -e fear -t filter_corpus

python filter_data-emotion.py -ip surprise_output/data -op task_1_output/surprise -e surprise -t filter_corpus

# Step 2: Using the output from Task 1, iterate through the folder of filtered JSON files. For each JSON file, concatenate the articles and add the combined text as a single row in the final CSV file for that emotion.
python filter_data-emotion.py -ip task_1_output/disgust -op task_2_output -e disgust -t json_to_csv  

python filter_data-emotion.py -ip task_1_output/love -op task_2_output -e love -t json_to_csv  

python filter_data-emotion.py -ip task_1_output/fear -op task_2_output -e fear -t json_to_csv  

python filter_data-emotion.py -ip task_1_output/surprise -op task_2_output -e surprise -t json_to_csv  


# Step 3: Concatenate all emotion CSVs into a single dataset
python concatenate_csv.py --input data/processed/disgust.csv data/processed/love.csv data/processed/fear.csv data/processed/surprise.csv --output data/final/all_emotions.csv
```

### Modeling

Training the stacking model that will output visualisation and trainned model:
```bash
python stacking_model.py --input-file "data/final/all_emotions.csv" --output-model "models/stacking_ensemble.pkl"
```

## Key Findings

Our exploration of emotion classification in olfactory contexts revealed several significant findings:

1. **Emotion-Specific Lexical Patterns**: Each emotion has distinctive vocabulary:
   - Love: "parfum" (perfume), "fleur" (flower), "rose"
   - Disgust: "saveur" (flavor), "fort" (strong), "désagréable" (unpleasant)
   - Fear: "danger", "feu" (fire), threat-related terms
   - Surprise: Shares vocabulary with love but with contextual differences

2. **OCR Noise Management**: Setting min_df=2 effectively reduced dimensionality from 78,000 to 33,000 features while preserving semantic information.

3. **Classifier Comparison**: SVM performed best individually (0.956 macro recall), while MultinomialNB and RandomForest showed strengths with specific emotions.

4. **Stacking Performance**: Our stacking approach achieved high accuracy (95.95%) but with limited improvement over the best base classifier, suggesting the need for more sophisticated ensemble strategies in future work.

## Project Pipeline

```
Raw Data Collection → Filtering → JSON to CSV Conversion → Concatenation → Preprocessing → Model Training → Evaluation
```

1. **Web Crawling**: Extract data from Odeuropa API
2. **Filtering & Formatting**: Keep only documents with target emotions and concatenate filtered articles into emotion specific rows for augmented documents.
4. **Concatenation**: Combine all emotion CSVs into a single dataset
5. **Preprocessing**: Handle OCR noise, stopwords, and feature reduction
6. **Model Training**: Implement stacking ensemble with base classifiers
7. **Evaluation**: Assess model performance with precision, recall, and F1-score

## Contributors

- **Jocelyn Zaruma**: Led data collection and corpus development, verified data quality, corpus statistics, text representation, and conducted preliminary data preprocessing.

- **Xingyu Chen**: Focused on model implementation, preprocessing pipeline, visualization, and evaluation


## Repository

[GitHub Repository](https://github.com/Yixian-ch/smell_emotion_classifier)