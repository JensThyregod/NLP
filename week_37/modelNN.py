from datasets import load_dataset
import Preprocessor as p
import numpy as np
import nltk
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import re
import torch.nn.functional as F
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from helpMe import get_ppl, convert_to_float, getList, Average, map_boolean, generate_embeddings, write_embeddings_to_csv
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from customDataset import CustomDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second hidden layer
        x = F.relu(self.fc3(x))  # Apply ReLU activation to the third hidden layer
        out = torch.sigmoid(self.fc4(x))  # Apply sigmoid activation to the output layer
        return out
