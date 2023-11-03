import numpy as np
import nltk
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from transformers import pipeline
from tqdm import tqdm
import re
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from bpemb import BPEmb



embedding_dim = 50
vocab_size = 10000

bpemb_bn = BPEmb(lang="bn", dim=embedding_dim, vs=vocab_size, add_pad_emb=True) # Bengali
bpemb_ar = BPEmb(lang="ar", dim=embedding_dim, vs=vocab_size, add_pad_emb=True) # Arabic
bpemb_id = BPEmb(lang="id", dim=embedding_dim, vs=vocab_size, add_pad_emb=True) # Indonesian

# Extract the embeddings and add an embedding for our extra [PAD] token
embeddings_ar = np.concatenate([bpemb_ar.emb.vectors, np.zeros(shape=(1,embedding_dim))], axis=0)
embeddings_bn = np.concatenate([bpemb_bn.emb.vectors, np.zeros(shape=(1,embedding_dim))], axis=0)
embeddings_id = np.concatenate([bpemb_id.emb.vectors, np.zeros(shape=(1,embedding_dim))], axis=0)
# Extract the vocab and add an extra [PAD] token
vocabulary_ar = bpemb_ar.emb.index_to_key + ['<pad>']
vocabulary_bn = bpemb_bn.emb.index_to_key + ['<pad>']
vocabulary_id = bpemb_id.emb.index_to_key + ['<pad>']


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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



def train_loop(model, model_name,train_loader, input_size = 768, output_size = 1, learning_rate = 0.001, num_epochs = 400):

    criterion = nn.BCELoss().to(device)  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    avg_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)

    
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)  # Squeeze the output to match targets shape
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
        avg_loss.append(average_loss)
    torch.save(model.state_dict(), model_name)
    return()
    
    
def train_rf(xs, ys):
    flattened_data = np.array([sub_array.flatten() for sub_array in xs])
    X_train, X_test, y_train, y_test = train_test_split(flattened_data, ys, test_size=0.1, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_classifier = RandomForestClassifier(random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, 
                               scoring='accuracy', cv=kf, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))


def labeller(row):
    """
    Tokenizes and encodes question and document, and creates
    corresponding labels.

    row: a pandas.core.series.Series (one row of dataframe)
    returns: (tokens, labels)
    """
    if row.language == 'indonesian':
        vocab = bpemb_id
    elif row.language == 'bengali':
        vocab = bpemb_bn
    elif row.language == 'arabic':
        vocab = bpemb_ar
    else:
        raise ValueError(f'Language not supported: {row.language}')

    # a: answer
    a_start = row.annotations.get('answer_start')[0]
    a = row.annotations.get('answer_text')[0]
    a_len = len(a) # answer char length

    # q: question
    q = row.question_text + ' [sep] '
    q_ids = vocab.encode_ids(q)
    q_ids_len = len(q_ids)

    # d: document
    d_pre = row.document_plaintext[:a_start] # document text before the answer
    d_pre_ids = vocab.encode_ids(d_pre)
    d_pre_ids_len = len(d_pre_ids)

    d_ans = row.document_plaintext[a_start:a_start+a_len] # answer in document text
    d_ans_ids = vocab.encode_ids(d_ans)
    d_ans_ids_len = len(d_ans_ids)

    d_post = row.document_plaintext[a_start+a_len:] # document text after answer
    d_post_ids = vocab.encode_ids(d_post)
    d_post_ids_len = len(d_post_ids)

    token_ids = torch.tensor(q_ids + d_pre_ids + d_ans_ids + d_post_ids)

    total_len = q_ids_len + d_pre_ids_len + d_ans_ids_len + d_post_ids_len
    pre_ans_len = q_ids_len + d_pre_ids_len
    labels = torch.zeros(total_len)
    labels[pre_ans_len : pre_ans_len + d_ans_ids_len] = torch.ones(d_ans_ids_len)

    return token_ids, labels


