## Feelings Finder: Classifying Emotions from Olfactory Contexts
## Quick Overview:
### Corpus

Excerpts from documents associated with olfactory contexts, specifically collected from the *Odeuropa Smell Browser* project. These excerpts contain words linked to various emotions.

#### Access Our Data

1. Script: [webcrawl.py](./webcrawl.py)
   Output: [data](https://drive.google.com/drive/folders/1KXq1Ulc01vNQN3bL8O-58WdWGYS39En4)

2. Script: [filter\_data-emotion.py](./filter_data-emotion.py)

   * Task 1 Output: [data](https://drive.google.com/drive/u/1/folders/1OjBwxKV4_BWFwru2o6rZrq99cm9g4mmI)
   * Task 2 Output: [data](https://drive.google.com/drive/u/1/folders/1R-TZRpRMys2TljBN6QPpOlkCGX9DIBCb)


### Classification Objective

We are working with excerpts containing words associated with the following emotions:

1. Love
2. Disgust
3. Fear
4. Surprise

### Strategy

#### Web Crawling and Scraping

1. `raw_data/`

   * Contains the raw webscraped documents
   * Preserved in case we need to extract more metadata later

2. `data/`

   * Filtered from `raw_data/` to include:

     * Metadata relevant to our task
     * The actual excerpts (texts)


#### Emotion Filtering

Since some texts may reflect multiple emotions, we selected emotion labels that were most distinct from one another. To refine the dataset, we kept only entries labeled with a single target emotion.

### Classifiers
