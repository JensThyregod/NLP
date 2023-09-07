import pandas as pd

class Preprocessor:
    
    def __init__(self, df):
        self.df = df

    def add_tokens_column(self):
        self.df['tokens'] = self.df['document_plaintext'].str.split()

    def clean_tokens(self, column_name='tokens'):
        def clean_token(token):
            for char in [',', ':', ';', '(', ')', '.']:
                token = token.replace(char, '')
            return token
        
        self.df[column_name] = self.df[column_name].apply(lambda tokens: [clean_token(token) for token in tokens])
