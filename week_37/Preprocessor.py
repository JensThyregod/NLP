import pandas as pd
import arabicstopwords.arabicstopwords as stp
from bltk.langtools import remove_stopwords, Tokenizer

class DataFramePreprocessor:
    
    def __init__(self, df, columns_to_tokenize, remove_stopwords=True):
        self.df = df.copy()
        self.columns_to_tokenize = columns_to_tokenize
        self.remove_stopwords = remove_stopwords
        self._add_tokens_columns()
        self._lowercase_tokens()
        self._clean_tokens()
        
        # Conditionally remove stopwords for the three languages
        if self.remove_stopwords:
            self._remove_arabic_stopwords()
            self._remove_bengali_stopwords()
            self._remove_indonesian_stopwords()
        
        self._count_all_tokens()

    def _add_tokens_columns(self):
        for col in self.columns_to_tokenize:
            self.df[f'{col}_tokens'] = self.df[col].str.split()

    def _lowercase_tokens(self):
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df[column_name].apply(lambda tokens: [token.lower() for token in tokens])

    def _count_all_tokens(self):
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.total_token_count = sum(self.df[column_name].apply(len))

    def _clean_tokens(self):
        def clean_token(token):
            for char in [',', ':', ';', '(', ')', '.', '?']:
                token = token.replace(char, '')
            return token
        
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df[column_name].apply(lambda tokens: [clean_token(token) for token in tokens])

    def _remove_arabic_stopwords(self):
        def filter_arabic_stopwords(tokens, lang):
            if lang == "arabic":
                return [token for token in tokens if not stp.is_stop(token)]
            return tokens
        
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df.apply(lambda row: filter_arabic_stopwords(row[column_name], row['language']), axis=1)

    def _remove_bengali_stopwords(self):
        def filter_bengali_stopwords(tokens, lang):
            if lang == "bengali":
                return remove_stopwords(tokens, level='moderate')  # adjust as needed
            return tokens
        
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df.apply(lambda row: filter_bengali_stopwords(row[column_name], row['language']), axis=1)

    def _remove_indonesian_stopwords(self):
        # load stopwords
        with open("tala-stopwords-indonesia.txt", "r") as f:
            stopword_list = [line.strip() for line in f]

        # Remove the stopwords from the tokens
        def filter_indonesian_stopwords(tokens, lang):
            if lang == "indonesian":
                return [token for token in tokens if token not in stopword_list]
            return tokens

        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df.apply(lambda row: filter_indonesian_stopwords(row[column_name], row['language']), axis=1)
