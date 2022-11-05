# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:02:47 2022

@author: Johann
"""
# Generate cosine matrix for emotion words and emojis
# Maybe some visuals??
import os, json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

with open('emoji ords.json','rt') as f:
    emoji_ords=json.load(f)

embedding_path = os.path.join('..', 'data','embeddings.csv')
emb_df = pd.read_csv(embedding_path, index_col=0)
emoji_emb = emb_df.iloc[-41:,:].to_numpy()

emotion_lex = pd.read_csv(os.path.join('..','data','hausa_emotion_lexicon.csv'))
emotion_words= emotion_lex.machine_translation_updated.unique()
english_words=[]
for word in emotion_words:
    english_words.append(emotion_lex.word[emotion_lex.machine_translation_updated==word].values[0])

tokenizer = AutoTokenizer.from_pretrained(os.path.join('..','models','emoji incl','Afriberta-base'))

emb_idx =[]
for word in emotion_words:
    emb_idx.append(tokenizer(word)['input_ids'][1:-1])

word_emb =[]
for i in emb_idx:
    word_emb.append(emb_df.iloc[i,:].mean(axis=0).values.tolist())
word_emb = np.array(word_emb)

cosines= cosine_similarity(emoji_emb, word_emb)

emoji_export =[]
n=5
for i, ordinal in enumerate(emoji_ords):
    row=[]
    row.append(chr(ordinal))
    top_n_tokens = cosines[i].argsort()[-n:]
    for token in top_n_tokens:
        hausa_word = emotion_words[token]
        row.append(hausa_word)
    for token in top_n_tokens:
        english_word = english_words[token]
        row.append(english_word)
    emoji_export.append(row)
pd.DataFrame(emoji_export, columns=['Emoji','Hausa word 1','Hausa word 2','Hausa word 3','Hausa word 4','Hausa word 5',
                                    'English word 1','English word 2','English word 3','English word 4','English word 5',]).to_csv('Emoji similarity export.csv')
        