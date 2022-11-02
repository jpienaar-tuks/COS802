# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:50:55 2022

I first wanted to try using tensorflow since I already had tensorflow setup to use my gpu
from the earlier COS801 project. However, I keep running into out-of-memory errors and all of
the huggingface guides and tutorials are absed on pytorch, so I'm gonna switch over.

@author: Johann
"""
import pandas as pd
import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import create_optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset

def preprocess_function(examples):
    return tokenizer(examples["before"], truncation=True, padding=True, return_tensors='tf')
    
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

def prepare_tf_dataset(df, tokenizer):
    hf_dataset = Dataset.from_pandas(df)
    tokenized_hf_dataset = hf_dataset.map(preprocess_function, batched=True)
    tokenizer_columns = tokenizer.model_input_names
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')
    tf_dataset = tokenized_hf_dataset.to_tf_dataset(columns=tokenizer_columns,
                                                    label_cols=['label'],
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    collate_fn=data_collator)
    return tf_dataset

def make_model(model_checkpoint, dataset_len):
    # Taking inspiration from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb#scrollTo=vzS60i7XrXMg
    # We'll be using create_optimizer from transformers
    tf_model = TFAutoModelForSequenceClassification.from_pretrained(model['checkpoint'], num_labels=LABELS,
                                                                    id2label=id2label, label2id=label2id)
    batches_per_epoch = dataset_len//batch_size
    total_train_steps = int(batches_per_epoch*num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_train_steps=total_train_steps, num_warmup_steps=0)
    tf_model.compile(optimizer=optimizer, metrics="accuracy")
    return tf_model
    

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

models = [{'checkpoint':'castorini/afriberta_base','name':'Afriberta'},
          {'checkpoint':'xlm-roberta-base', 'name':'xlm-roberta-base'},
          {'checkpoint':'google/rembert','name':'RemBERT'}]
LABELS=3
batch_size=4
num_epochs = 3

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

history=dict()
for model in models:
    tf.keras.backend.clear_session()
    tokenizer = AutoTokenizer.from_pretrained(model['checkpoint'], num_labels=LABELS,
                                          id2label=id2label, label2id=label2id)
    tf_train = prepare_tf_dataset(train_df, tokenizer)
    tf_validation = prepare_tf_dataset(validation_df, tokenizer)
    tf_model = make_model(model['checkpoint'], len(train_df))
    tf_model.layers[0].trainable=False
    history[model['name']] = tf_model.fit(tf_train, validation_data=tf_validation, epochs=num_epochs)
    # TODO: Save models after fitting. Probably fairly big, but probably worth it given the long training times
    


# data=tf_model.prepare_tf_dataset(tokenized_dataset_train, batch_size=32, shuffle=True, tokenizer=tokenizer) # Alternative??

# model_test = 


# training_args = TrainingArguments(
#     output_dir=os.path.join('..','results',model['name']),
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     report_to = "none"
# )

# trainer = Trainer(
#     model=model_test,
#     args=training_args,
#     train_dataset=tokenized_dataset_train,
#     eval_dataset=tokenized_dataset_test,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics = compute_metrics,
# )

# trainer.train()

