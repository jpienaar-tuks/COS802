import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = os.path.join('..','models','emoji incl','Afriberta-base')
model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

emb = model.get_input_embeddings()
embeddings = emb(torch.LongTensor(range(emb.num_embeddings)))
embeddings_numpy = embeddings.detach().numpy()
df=pd.DataFrame(embeddings_numpy)
df.to_csv(os.path.join('..','data','embeddings.csv'))

