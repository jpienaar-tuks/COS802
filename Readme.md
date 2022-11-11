# COS802
Project is to examine the impact of emoji's on a 
sentiment analysis task for a low resource language (Hausa).

Project will be using the Afrisenti-Semeval dataset as well as
several multilingual transformer models:
* castorini/afriberta_small
* castorini/afriberta_base
* xlm-roberta-base 

## src directory
* COS802 PyTorch.py - Main python script to train the three models for each of the three scenarios. Produces output in csv files for model performance and training times. Also saves models in the 'models' directory (\*.bin files excluded from this zip due to size)
* COS802 TensorFlow experiment.py - An early abandoned experiment using tensorflow as a backend
* extract embeddings.py - Code to extract the embeddings from the finetuned models. Generates a large csv file in the 'data' folder
* emoji cosine similarity.py - code to examine the cosine similarity of the learned embeddings for emojis and the Hausa emotion lexicon
* tSNE visual.py - generates a tSNE visualisation for the learned emoji embeddings

## data directory
* ha_train.tsv - Main dataset, taken from [Afrisent-Semeval-2023](https://github.com/afrisenti-semeval/afrisent-semeval-2023)
* hausa_emotion_lexicon - Used to evaluate learned emoji embeddings against the Hausa emotion lexicon provided by the NaijaSenti repo (https://github.com/shmuhammad2004/NaijaSenti/blob/main/data/translated_emotion_lexicon/hausa_emotion_lexicon.csv)

