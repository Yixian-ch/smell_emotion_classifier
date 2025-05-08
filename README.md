## Feelings Finder: Classifying Emotions from Olfactory Contexts

## Quick Overview:

### Corpus

Excerpts are drawn from documents associated with olfactory contexts, specifically collected from the *Odeuropa Smell Browser* project. These excerpts contain words linked to various emotions.

#### Access Our Data

1. Script: [webcrawl.py](./webcrawl.py)

   - Output: [data](https://drive.google.com/drive/folders/1KXq1Ulc01vNQN3bL8O-58WdWGYS39En4)

   **Organization:**
   - `raw_data/`
     - Contains the raw web-scraped documents
     - Preserved in case we need to extract more metadata later

   - `data/`
     - Filtered from `raw_data/` to include:

       - Metadata relevant to our task
       - The actual excerpts (texts)

2. Script: [filter\_data-emotion.py](./filter_data-emotion.py)

   - Task 1 Output: [data](https://drive.google.com/drive/u/1/folders/1OjBwxKV4_BWFwru2o6rZrq99cm9g4mmI)
   - Task 2 Output: [data](https://drive.google.com/drive/u/1/folders/1R-TZRpRMys2TljBN6QPpOlkCGX9DIBCb)

   **Organization:**
   
   - `task_1_output/`

     - Contains subfolders for each target emotion; each subfolder holds JSON files representing pages from the API responses
     - We filter out only the articles that have the target emotion label

   - `task_2_output/`

     - Uses the outputs from Task 1 to augment the text for our data

### Classification Objective

We are working with excerpts containing words associated with the following emotions:

1. Love
2. Disgust
3. Fear
4. Surprise

#### Emotion Filtering

Since some texts may reflect multiple emotions, we selected emotion labels that are most distinct from one another. To refine the dataset, we kept only entries labeled with a single target emotion.

### Classifiers

