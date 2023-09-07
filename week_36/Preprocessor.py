import pandas as pd
import arabicstopwords.arabicstopwords as stp
from bltk.langtools import remove_stopwords, Tokenizer

class DataFramePreprocessor:
    
    def __init__(self, df):
        self.df = df.copy()
        self._add_tokens_column()
        self._clean_tokens()
        self._remove_arabic_stopwords()
        self._remove_bengali_stopwords()

    def _add_tokens_column(self):
        self.df['tokens'] = self.df['document_plaintext'].str.split()

    def _clean_tokens(self, column_name='tokens'):
        def clean_token(token):
            for char in [',', ':', ';', '(', ')', '.']:
                token = token.replace(char, '')
            return token
        
        self.df[column_name] = self.df[column_name].apply(lambda tokens: [clean_token(token) for token in tokens])

    def _remove_arabic_stopwords(self, column_name='tokens'):
        def filter_arabic_stopwords(tokens, lang):
            if lang == "Arabic":
                return [token for token in tokens if not stp.is_stop(token)]
            return tokens
        
        self.df[column_name] = self.df.apply(lambda row: filter_arabic_stopwords(row[column_name], row['language']), axis=1)

    def _remove_bengali_stopwords(self, column_name='tokens'):
        def filter_bengali_stopwords(tokens, lang):
            if lang == "Bengali":
                return remove_stopwords(tokens, level='moderate')  # You can change the level as needed
            return tokens
        
        self.df[column_name] = self.df.apply(lambda row: filter_bengali_stopwords(row[column_name], row['language']), axis=1)

