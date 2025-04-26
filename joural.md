# Emotion_classification
This projet intends to build a multinominal classifier to take a text as input and predict its emotion. To do this, we have firstly choosen the `Odeuropa Smell Explorer` a project using license as CC BY 4.0.
Steps to train a classification model.
- Collecing data 
- Preprocessing data
- Presenting data
- Training model

## Collecting data 
Jojo has written a script to scrawb the data then save them as json files in two directorys. One is data, another is raw, what we interest here is the data direcotry. Because we need the `exerpt` to constitute our corpus as training data, and the `emotions` as label.
### Fomring collected data
As raw data be collected, we will extract data from is and save them to a floder called data. 
The structure of files: a json file with only one key which is nameds `articles`. It contains a list of dict where each dict is an article of the current file. The dictionary contains `doc_url` (the url where excerpt descriptions are collected), `emotions` (emotions of the current article), `excerpt` (description), `id` (the order how we collect the data), `title` (the title showed in the current article).

## Preprocessing data
- Before preprocessing, some articles may have not only one emotion, that make them ambiguous. So we choose to extract those articles containing only our target emotions. For example, for love, the key `emotions` can only contain `love` nothing else.
- Then write a script to transform our json files into a single csv, where each row is a json file.
- Use `nltk` to do preprocessing: tokenization, remove punktuations, remove French stopwords, and the lemmatization is opetional.
- Our data has an interesting thing which is for the French apostroph `'`, it is surrounded by space. This is not normal, cause the apostroph is used to link pronom like `je, tu` which are determined by a vowel when they are used with words starting with a vowel, pronom's last character will be replaced by the apostroph `'`. Like `je aime -> j'aime` to facilitate the pronunciation. While in our corpus, apostroph is surrounded by space like `j ' aime`.  So to fix it, we've use a regular expression during our preprocessing.

## Split data
- use `sklearn.modelselection` `train_test_split` to split our data into training and testing data. Use argument `test_size` to decide how to split 

## Representating data
- use `gensim` word2vec model to transform words into vectors representation, interpretable by computer.

## Models
- use sklearn models like randomforest and svm to test our data's quality. And use f-score to measure model's performance.