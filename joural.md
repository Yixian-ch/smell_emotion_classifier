# Emotion_classification
This projet intends to build a multinominal classifier to take a text as input and predict its emotion. To do this, we have firstly choosen the `Odeuropa Smell Explorer` a project using license as `CC BY 4.0`.
Steps to train a classification model.
- Collecing data 
- Preprocessing data
- Presenting data
- Training model

## Collecting data 
- Jojo has written a script to scrawb the data then save them as json files in two directorys. One is data, another is raw, what we interest here is the data direcotry. Because we need the `exerpt` to constitute our corpus as training data, and the `emotions` as label.
### Fomring collected data
- As raw data be collected, we will extract data from is and save them to a floder called data. 
- The structure of files: a json file with only one key which is nameds `articles`. It contains a list of dict where each dict is an article of the current file. The dictionary contains `doc_url` (the url where excerpt descriptions are collected), `emotions` (emotions of the current article), `excerpt` (description), `id` (the order how we collect the data), `title` (the title showed in the current article).
- After building our models, we've takend a look at what are the mots important words helping a classifier to predict an emotion. Two things have been discorved, one is that the model has learned how to distinguish emotion by lexical information like humain. For example, for `fear`, the top 10 features for prediction are `danger`, `odeur`, `odeur danger`, `feu`, `mort`... For us, those words are also related to fear. Another thing is, in the top n features for predict, there are still stopwords like `comme`, `où`, `tout`, `cette`, `odeur`, `plus`, `sans`, `dont` and `parfum`, `parfums`. So we need to update our stopwords list and do lemmatization.

## Preprocessing data
- Before preprocessing, some articles may have not only one emotion, that make them ambiguous. So we choose to extract those articles containing only our target emotions. For example, for love, the key `emotions` can only contain `love` nothing else. 
- After the emotion slection, for `love` we hanve 798 json files, for `fear`, we have 491 json files, for `disgust` we have 801 files and for `surprise` we have 504 files.
- To build a list of corpus, we were wondering if we combine texts by source, if texts have same url, we combine them together. But after testing, we've discorved that after cleaning process, we do not have enough texts for a same url. (because the cleaning keeps texts only containing one single emotion. So for texts come from a same url, only those have one single emotion will be saved. In the end, text are globally from different url) So, we choose to extract text from json files to concatate them as a corpus file by file. Then write trasnformed json files into a single csv, where each row is a the concatated corpus. Because, even for those texts extracetd from the same url, they may not from the same paragraph or the same section. The global meaning is already missing, so it's ok if we concatate texts from different url and our classification models do have not bay performance on them.
- Use `nltk` to do preprocessing: tokenization, remove punktuations, remove French stopwords, and the lemmatization is opetional.
- Our data has an interesting thing which is for the French apostroph `'`, it is surrounded by space. This is not normal, cause the apostroph is used to link pronom like `je, tu` which are determined by a vowel when they are used with words starting with a vowel, pronom's last character will be replaced by the apostroph `'`. Like `je aime -> j'aime` to facilitate the pronunciation. While in our corpus, apostroph is surrounded by space like `j ' aime`.  So to fix it, we've use a regular expression during our preprocessing.
- 

## Split data
- use `sklearn.modelselection` `train_test_split` to split our data into training and testing data. Use argument `test_size` to decide how to split 

## Representating data
- use `gensim` word2vec model to transform words into vectors representation, interpretable by computer.

## Models
- use sklearn models like randomforest and svm to test our data's quality. And use f-score to measure model's performance.