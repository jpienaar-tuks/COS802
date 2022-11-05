# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:34:41 2022

@author: Johann
"""
import os, json
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open('emoji ords.json','rt') as f:
    emoji_ords=json.load(f)

embedding_path = os.path.join('..', 'data','embeddings.csv')
emb_df = pd.read_csv(embedding_path, index_col=0)
emoji_emb = emb_df.iloc[-41:,:].to_numpy()


#------------------------------------------------------------------------------
# t-SNE visualisation of learned emoji embeddings
#------------------------------------------------------------------------------
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=1500)

labels=[chr(i) for i in emoji_ords]
values_emoji_tsne = tsne_model.fit_transform(emoji_emb)

x=[]
y=[]
for i,j in values_emoji_tsne:
    x.append(i)
    y.append(j)
    
fig= plt.figure()
for i in range(len(emoji_ords)):
    plt.scatter(x[i],y[i], c='white')
    plt.annotate(labels[i], xy=(x[i],y[i]), fontname='Segoe UI Emoji', fontsize='large')
plt.show()
