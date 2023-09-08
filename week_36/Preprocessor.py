import pandas as pd
import arabicstopwords.arabicstopwords as stp
from bltk.langtools import remove_stopwords, Tokenizer

class DataFramePreprocessor:
    
    def __init__(self, df):
        self.df = df.copy()
        self._add_tokens_column()
        self._lowercase_tokens()
        self._clean_tokens()
        self._remove_arabic_stopwords()
        self._remove_bengali_stopwords()
        self._remove_indonesian_stopwords()
        self._count_all_tokens()

    def _add_tokens_column(self):
        self.df['tokens'] = self.df['document_plaintext'].str.split()

    def _lowercase_tokens(self, column_name='tokens'):
        self.df[column_name] = self.df[column_name].apply(lambda tokens: [token.lower() for token in tokens])

    def _count_all_tokens(self, column_name='tokens'):
        self.total_token_count = sum(self.df[column_name].apply(len))

    def _clean_tokens(self, column_name='tokens'):
        def clean_token(token):
            for char in [',', ':', ';', '(', ')', '.']:
                token = token.replace(char, '')
            return token
        
        self.df[column_name] = self.df[column_name].apply(lambda tokens: [clean_token(token) for token in tokens])

    def _remove_arabic_stopwords(self, column_name='tokens'):
        def filter_arabic_stopwords(tokens, lang):
            if lang == "arabic":
                return [token for token in tokens if not stp.is_stop(token)]
            return tokens
        
        self.df[column_name] = self.df.apply(lambda row: filter_arabic_stopwords(row[column_name], row['language']), axis=1)

    def _remove_bengali_stopwords(self, column_name='tokens'):
        def filter_bengali_stopwords(tokens, lang):
            if lang == "bengali":
                return remove_stopwords(tokens, level='moderate')  # <--- kan justeres
            return tokens
        
        self.df[column_name] = self.df.apply(lambda row: filter_bengali_stopwords(row[column_name], row['language']), axis=1)

    def _remove_indonesian_stopwords(self, column_name='tokens'):
        ## load stopwords
        with open("tala-stopwords-indonesia.txt", "r") as f:
            stopword_list = [line.strip() for line in f]

        # Remove the stopwords from the tokens
        def filter_indonesian_stopwords(tokens, lang):
            if lang == "indonesian":
                return [token for token in tokens if token not in stopword_list]
            return tokens

        self.df[column_name] = self.df.apply(lambda row: filter_indonesian_stopwords(row[column_name], row['language']), axis=1)
