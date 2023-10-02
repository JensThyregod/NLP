import numpy as np
import nltk
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from transformers import pipeline
import torch
from tqdm import tqdm
import re
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model
import torch
import pandas as pd

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')



def map_boolean(value):
    return 1 if value else 0


def Average(lst):
    return sum(lst) / len(lst)



def get_ppl(lang, strng):
    #ppl_return = []
    f = open(strng+'.csv', "w")
    
    
    for i in range(len(lang)):
        inputs = tokenizer(lang['question_text'].iloc[i], return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        loss_str = str(torch.exp(loss))
        extracted_float = re.search(r'[-+]?\d*\.\d+|\d+', loss_str).group()
        f.write(extracted_float+',')
    f.close()

    
def convert_to_float(original_list):
    float_list = []
    for item in original_list:
        try:
            float_value = float(item)
            float_list.append(float_value)
        except ValueError:
            # Handle the case where the conversion to float is not possible
            pass
    return float_list 


def getList(path):

    csv_file_path = path
    data_list = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data_list = []
        for row in csv_reader:
            data_list.append(row)
    return data_list




def generate_embeddings(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)  
    return embeddings.numpy()
#Uses mean-pooling

def write_embeddings_to_csv(texts, csv_file):
    embeddings_list = []
    for text in texts:
        embeddings = generate_embeddings(text)
        embeddings_list.append(embeddings)
    
    df = pd.DataFrame({'text': texts, 'embeddings': embeddings_list})
    df.to_csv(csv_file, mode='a', header=False, index=False)