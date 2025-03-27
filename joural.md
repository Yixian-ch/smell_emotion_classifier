## Processing the json data
### Data structure
Our collected data contains four keys, what we are interested is the `results` that has the emotion and description text. The problem is what kind of data we need to extract. For example, for the key `results` its value is a list of dict where each dict represents one post, 