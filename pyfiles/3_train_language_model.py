import os
import argparse
import requests
import re
import json
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import transformers
import torch

from tqdm import tqdm, trange
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

webhook_url = 'https://discordapp.com/api/webhooks/759743974671384608/wl_A7P2ap8DUBB-0OjHD_HxFeblPE95asDilHrZXCWK3UuxIzBkXHWmhRTIuYJex4s8r'

def send_message(text: str):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'content': text})
    requests.post(url=webhook_url, data=payload, headers=headers)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, help='Name of the city to download articles for.')
    parser.add_argument('--batch_size', type=int, help='Size of the each training batch.')
    parser.add_argument('--label_column', type=str, help='column of the df which is the label')
    parser.add_argument('--model', type=str, default='bert-base-uncased' ,help='transformer model to use')
    parser.add_argument('--save_path', type=str, help='path to the folder to save the articles to.')

    return parser.parse_args()



def load_articles(path: str) -> pd.DataFrame:
    
    """
    Loads all the articles from desired folder.

    Parameters
    ----------
    path: str
        Path of the city to load articles from.

    Returns
    -------
    df: pd.DataFrame
        dataframe with the articles and metadata.
    """
    df = pd.DataFrame()
    
    cbs = os.listdir(path)
    for cb in tqdm(cbs, leave=False):

        cb_path = os.path.join(path, cb)
        neighborhoods = os.listdir(cb_path)
        
        for neighborhood in neighborhoods:
            neighborhood_path = os.path.join(cb_path, neighborhood)
            articles = os.listdir(neighborhood_path)
            
            for article in articles:
                article_path = os.path.join(neighborhood_path, article)
                with open (article_path, 'r', encoding='utf-8') as f:
                    article = f.read().lower()
                series = pd.Series({'cb': cb, 'neighborhood': neighborhood, 'articles': article})
                df = df.append(series, ignore_index=True)
                
    return df

def encode_df(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    
    """
    Encodes the df labels using Label encoder and reuturns the new encoded df, and a labels dict.

    Parameters
    ----------

    df: pd.DataFrame
        Articles dataframe.

    label_column: str
        Which column in the dataframe would be used as the label from the classification task.

    Returns
    -------

    df: pd.DataFrame
        the new dataframe which label_column column is encoded using label encoder.

    id2label: Dict[int, str]
        Dictionary mapping integers to labels
    """
    lbe = LabelEncoder().fit(df[label_column].tolist())
    id2label = dict(zip(range(len(lbe.classes_)), lbe.classes_))
    
    df[label_column] = lbe.transform(df[label_column])
    
    return df, id2label

def get_transformer_model(id2label: Dict[int, str], pre_trained: str) -> Tuple[transformers.PreTrainedTokenizer, transformers.PreTrainedModel]:
    """
    Generates a transormfer model for a specific city based on the classification task which is defined by the column parameter.

    Parameters
    ----------
    id2label: Dict[int, str]
        Dictionary mapping between integers and label names.

    pre_trained: str
        path to the pre-trained model to load.

    Returns
    -------
    tokenizer: transformers.PreTrainedTokenizer
        transormfer tokenizer.

    model: transformers.PreTrainedModel
        pre_trained models with a final head with num_labels equal to the amount of unique items in label_column for the desired city.
    """

    tokenizer = AutoTokenizer.from_pretrained(pre_trained)
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained,
                                                               num_labels=len(id2label.keys()),
                                                               return_dict=True,
                                                               id2label=id2label,
                                                               label2id={v: k for k, v in id2label.items()})

    return tokenizer, model

def train_model(df: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer,
                model: transformers.PreTrainedModel, steps: int, batch_size: int, save_path:str) -> None:
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=5e-5)

    losses = []
    for step in trange(steps):
    
        optim.zero_grad()

        sample = df.sample(batch_size)

        X = sample['articles'].tolist()
        y = sample['labels'].tolist()
        
        inputs = tokenizer(X, return_tensors='pt', padding=True, truncation=True)
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = torch.tensor(y).unsqueeze(1).to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        losses.append(loss)
        
        if (step + 1) % 100 == 0:
            print(f'Step: {step + 1} Loss: {sum(losses)/len(losses)}')
            send_message(f'Step: {step + 1} Loss: {sum(losses)/len(losses)}')
            losses = []
            
        loss.backward()
        optim.step()

    model.save_pretrained(save_path)

def clear_names(article: str) -> str:
    """
    Clear any city or neighborhood indication in the text to prevent leakage.
    
    Parameters
    ----------
    article: str
        Article from the dataset to filter from.
    
    Returns
    -------
    article: str
        The same article after getting cleared by regex.
    """
    regex = f'({"|".join(bad_words)})'
    article = re.sub(regex, "token", article)
    article = re.sub(r'\s+', "", article)
    return article

def main():

    parser = parse_args()

    print('Loading Data')
    news_articles_df = load_articles(f'../data/articles/{parser.city}')
    
    print('Ecoding df')
    df, id2label = encode_df(news_articles_df, parser.label_column)

    print('Getting Model')
    tokenizer, model = get_transformer_model(id2label, pre_trained=parser.model)

    train_data = df[['articles', parser.label_column]]
    train_data.columns = ['articles', 'labels']

    steps = (train_data.shape[0]//parser.batch_size) * 1
    print('Training the Model')
    train_model(df=train_data, tokenizer=tokenizer, model=model, steps=steps, batch_size=parser.batch_size, save_path=parser.save_path)

if __name__ == "__main__":
    main()
