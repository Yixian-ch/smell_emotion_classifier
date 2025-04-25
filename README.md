## Feelings Finder: Classifying emotions from olfactory contexts

### Corpus:
Excerpts from documents that were collected based on their association with olfactory contexts, specifically from the Odeuropa Smell Browser project. These excerpts contain words linked to different emotions.

### Classification objective:
We are working with excerpts that have words associated with the following emotions:
1. love
2. sadness
3. fear
4. trust

### Strategy:

#### Webcrawl and Webscrape:
- We saved the raw webscrape
    - in order to be able to go back to the raw webscrape for more metadata if needed.
    - Filtered the raw data for:
        - the metadata pertinent to our task
        - and the excerpts (texts)

#### Emotion:
Since the dataset includes texts that could reflect more than one emotion, we picked emotions that felt the most distinct from one another. To further clean/filter, we just kept data that only had this exact emotion.