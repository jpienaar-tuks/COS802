# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:50:55 2022

I first wanted to try using tensorflow since I already had tensorflow setup to use my gpu
from the earlier COS801 project. However, I keep running into out-of-memory errors and all of
the huggingface guides and tutorials are absed on pytorch, so I'm gonna switch over.

@author: Johann
"""
import pandas as pd
import os, gc, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import AdamW, get_constant_schedule_with_warmup
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
    """ Some of the model's I'll be evaluating already contains emoji in their tokenizer. I'll therefore need to
    strip the emoji's for the base case evaluations. Written Hausa does use some diacritics and possibly arabic 
    symbols, so I'll define emoji as anything where ord(c) returns a value above 9000"""
    s=[]
    for c in sentence:
        if ord(c) >=9000:
            continue
        s.append(c)
    return ''.join(s)

def make_and_train_model(model_name, checkpoint, scenario, classifier_only, train_ds, validation_ds, resize):
    model_pt = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=LABELS)
    if resize:
        model_pt.resize_token_embeddings(resize)
    ## Parameters to be passed to Trainer https://huggingface.co/docs/transformers/main_classes/trainer
    training_args = TrainingArguments(output_dir=os.path.join('..','results', scenario, model_name),
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_epochs,
                                      gradient_accumulation_steps=16,
                                      weight_decay=0.01,
                                      report_to = "none"
                                      )
    if classifier_only:
        optimizer = AdamW(model_pt.classifier.parameters()) # Only want to train the classifier head
        scheduler = get_constant_schedule_with_warmup(optimizer,0)
        opt_kwargs = {"optimizers": (optimizer, scheduler)}
    else:
        opt_kwargs ={}
    
    trainer = Trainer(model=model_pt,
                      args=training_args,
                      train_dataset=train_ds,
                      eval_dataset=validation_ds,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics = compute_metrics,
                      **opt_kwargs
                      )
    results = trainer.train()
    trainer.save_model(os.path.join('..','models',scenario,model_name))
    performance =trainer.evaluate()
    return results, performance

def vocab_stats(df, tokenizer):
    token2id = tokenizer.vocab
    id2token = {value:key for key, value in token2id.items()}
    token_counts = {idx: 0 for idx in id2token.keys()}
    for idx, row in df.iterrows():
        ids = tokenizer(row['before'])['input_ids']
        for i in ids:
            try:
                token_counts[i]+=1
            except KeyError:
                token_counts[i]=1
                print(f"Potential OOV token ID: {i}")
    return token_counts, id2token

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

performance=dict()
results=dict()
scenario = 'classifier only'
for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model['checkpoint'])
    tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
    tokenized_dataset_validation = dataset_validation.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    r, p = make_and_train_model(model['name'], model['checkpoint'], scenario, True, 
                         tokenized_dataset_train, tokenized_dataset_validation, False)
    performance[model['name']]=p
    results[model['name']]=r
    # Just some memory management, since I seem to occasionally seem to run out 
    # and old model data can be cleared
    torch.cuda.empty_cache()
    gc.collect()

pd.DataFrame(performance).to_csv(f'{scenario} results.csv')
pd.DataFrame({key: results[key][2] for key in results.keys()}).to_csv(f'{scenario} training report.csv')

#------------------------------------------------------------------------------
# Next scenario:
# Now that we have trained the classifier heads, I want to examine the model
# performance after language adaptive finetuning. I'll reload the models from
# the previous step, but this time I'll train the entire model. I'll also reuse
# the no_emoji dataset from the previous step
#------------------------------------------------------------------------------

performance=dict()
results=dict()
scenario = 'LAFT'
for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model['checkpoint'])
    tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
    tokenized_dataset_validation = dataset_validation.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    checkpoint = os.path.join('..','models','classifier only',model['name'])
    
    r, p = make_and_train_model(model['name'], checkpoint, scenario, False, 
                         tokenized_dataset_train, tokenized_dataset_validation, False)
    performance[model['name']]=p
    results[model['name']]=r
    # Just some memory management, since I seem to occasionally seem to run out 
    # and old model data can be cleared
    torch.cuda.empty_cache()
    gc.collect()

pd.DataFrame(performance).to_csv(f'{scenario} results.csv')
pd.DataFrame({key: results[key][2] for key in results.keys()}).to_csv(f'{scenario} training report.csv')

#------------------------------------------------------------------------------
# Last scenario:
# We'll be adding the emojis back in to examine the impact that it has on the 
# sentiment analysis task. The xlm-roberta-base tokenizer already has emojis
# in its vocab, but for the afriberta models we'll have to add it in. 
#------------------------------------------------------------------------------

train_df, validation_df = train_test_split(hausa_df, test_size=0.1)

dataset_train = Dataset.from_pandas(train_df)
dataset_validation = Dataset.from_pandas(validation_df)

with open('emoji ords.json','rt') as f:
    emoji_ords=json.load(f)

performance=dict()
results=dict()
scenario = 'emoji incl'
for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model['checkpoint'])
    if 'afriberta' in model['checkpoint']:
        added = tokenizer.add_tokens([chr(i) for i in emoji_ords])
        print(f'Added {added} tokens to tokenizer')
    tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
    tokenized_dataset_validation = dataset_validation.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    checkpoint = os.path.join('..','models','LAFT',model['name'])
    resize = len(tokenizer)
    r, p = make_and_train_model(model['name'], checkpoint, scenario, False, 
                         tokenized_dataset_train, tokenized_dataset_validation, resize)
    performance[model['name']]=p
    results[model['name']]=r
    # Just some memory management, since I seem to occasionally seem to run out 
    # and old model data can be cleared
    torch.cuda.empty_cache()
    gc.collect()

pd.DataFrame(performance).to_csv(f'{scenario} results.csv')
pd.DataFrame({key: results[key][2] for key in results.keys()}).to_csv(f'{scenario} training report.csv')