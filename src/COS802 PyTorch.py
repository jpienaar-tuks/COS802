# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:50:55 2022

I first wanted to try using tensorflow since I already had tensorflow setup to use my gpu
from the earlier COS801 project. However, I keep running into out-of-memory errors and all of
the huggingface guides and tutorials are absed on pytorch, so I'm gonna switch over.

@author: Johann
"""
import pandas as pd
import os, gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
import torch

def preprocess_function(examples):
    return tokenizer(examples["before"], truncation=True, padding=True, return_tensors='pt')
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def strip_emojis(sentence):
    """ Some of the model's I'll be evaluating aleardy contains emoji in their tokenizer. I'll therefore need to
    strip the emoji's for the base case evaluations. Written Hausa does use some diacritics and possibly arabic 
    symbols, so I'll define emoji as anything where ord(c) returns a value above 9000"""
    s=[]
    for c in sentence:
        if ord(c) >=9000:
            continue
        s.append(c)
    return ''.join(s)

def vocab_stats(df, tokenizer):
    # TODO
    pass

#------------------------------------------------------------------------------
# Just gonna setup some variables, read the main DataFrame, etc.
#------------------------------------------------------------------------------

models = [{'checkpoint':'castorini/afriberta_small','name':'Afriberta-small'},
          {'checkpoint':'castorini/afriberta_base','name':'Afriberta-base'},
          {'checkpoint':'xlm-roberta-base', 'name':'xlm-roberta-base'},
          #{'checkpoint':'google/rembert','name':'RemBERT'} # Too large :sad:
          ]
LABELS=3
batch_size=4
num_epochs = 2

hausa_df = pd.read_csv(os.path.join(os.getcwd(),'..','data','ha_train.tsv'),sep='\t',index_col='ID')

# Looks like the .to_tf_dataset method on huggingface datasets have a problem with string labels? 
# Maybe need to OneHotEncode the label feature, for now I'll just enumerate them
label2id={'negative':0, 'neutral':1, 'positive':2}
id2label={value:key for key, value in label2id.items()}
hausa_df['label']=hausa_df['label'].map(label2id)

#------------------------------------------------------------------------------
# Let's begin with some scenarios:
# First, I want to see the baseline performance of the language models without 
# emojis. We'll freeze the base model and just train the SequenceClassification 
# head. We'll also strip off the emojis in the source text so everyting is equal
#------------------------------------------------------------------------------

hausa_no_emoji = hausa_df.copy(deep=True)
hausa_no_emoji.before = hausa_no_emoji.before.map(strip_emojis)
train_df, validation_df = train_test_split(hausa_no_emoji, test_size=0.1)

dataset_train = Dataset.from_pandas(train_df)
dataset_validation = Dataset.from_pandas(validation_df)

history=dict()
results=dict()
for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model['checkpoint'])
    tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
    tokenized_dataset_validation = dataset_validation.map(preprocess_function, batched=True)
    model_pt = AutoModelForSequenceClassification.from_pretrained(model['checkpoint'], num_labels=LABELS)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    ## Parameters to be passed to Trainer https://huggingface.co/docs/transformers/main_classes/trainer
    training_args = TrainingArguments(output_dir=os.path.join('..','results','classifier only',model['name']),
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_epochs,
                                      gradient_accumulation_steps=16,
                                      weight_decay=0.01,
                                      report_to = "none"
                                      )
    optimizer = AdamW(model_pt.classifier.parameters()) # Only want to train the classifier head
    scheduler = get_constant_schedule_with_warmup(optimizer,0)
    trainer = Trainer(model=model_pt,
                      args=training_args,
                      train_dataset=tokenized_dataset_train,
                      eval_dataset=tokenized_dataset_validation,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics = compute_metrics,
                      optimizers = (optimizer,scheduler)
                      )
    results[model['name']]=trainer.train()
    trainer.save_model(os.path.join('..','models','classifier only',model['name']))
    history[model['name']]=trainer.evaluate()
    # Just some memory mangement, since I seem to occasionally seem to run out 
    # and old model data can be cleared
    del trainer, model_pt
    torch.cuda.empty_cache()
    gc.collect()
    