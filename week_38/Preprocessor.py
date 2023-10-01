import pandas as pd

class DataFramePreprocessor:
    def __init__(self, df, 
                 columns_to_tokenize, 
                 remove_stopwords=True, 
                 remove_punctuation=True, 
                 add_special_tokens=True,
                 count=True):
        
        self.df = df.copy()
        self.columns_to_tokenize = columns_to_tokenize
        self.remove_stopwords = remove_stopwords
        
        self._add_tokens_columns()
        
        
        self._split()
        self._lowercase_tokens()
        
        if not remove_punctuation:
            print("Replacing punctuation with tokens...")
            self._tokenize_punctuation()
        else:
            self._remove_punctuation()

        # Add special tokens
        if add_special_tokens:
            self._add_special_tokens()

        
        if self.remove_stopwords:
            self._remove_arabic_stopwords()
            self._remove_bengali_stopwords()
            self._remove_indonesian_stopwords()
        
        self._add_eos()
        
        if count:
            self._count_all_tokens()

    def _add_tokens_columns(self):
        """Split text on whitespace and return result in new column"""
        for col in self.columns_to_tokenize:
            self.df[f'{col}_tokens'] = self.df[col]
    
    def _split(self):
        for col in self.columns_to_tokenize:
            self.df[f'{col}_tokens'] = self.df[col].str.split()

    def _lowercase_tokens(self):
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df[column_name].apply(lambda tokens: [token.lower() for token in tokens])

    def _remove_punctuation(self):
        def clean_token(token):
            for char in [',', ':', ';', '(', ')', '.', '?']:
                token = token.replace(char, '')
            return token
        
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df[column_name].apply(lambda tokens: [clean_token(token) for token in tokens])
    
    def _tokenize_punctuation(self):
        def tokenize(text):
            """replace punctuation with tokens"""
            replacement_dict = {
                ',': '<COM>',
                '.': '<PUN>',
                ':': '<COL>',
                ';': '<SEM>',
                '(': '<LPA>',
                ')': '<RPA>',
                '?': '<QUE>',
                '!': '<EXC>',
                '"': '<QUO>',
                '\'': ''
            }
            
            # Split tokens that end with punctuation
            split_tokens = []
            for token in text:
                for punct, replacement in replacement_dict.items():
                    if token.endswith(punct):
                        split_tokens.extend([token.rstrip(punct), punct])
                        break
                else:
                    split_tokens.append(token)
            
            # Replace punctuation with corresponding tokens
            tokens = [replacement_dict.get(token, token) for token in split_tokens]
            
            return tokens

        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df[column_name].apply(tokenize)

    def _add_special_tokens(self):
        """Add <Q>, </Q>, <D> and </D> for question_text column and document_paintext respectivly
        """
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            if col == 'question_text':
                self.df[column_name] = self.df[column_name].apply(lambda tokens: ['<Q>'] + tokens + ['</Q>'])
            elif col == 'document_plaintext':
                self.df[column_name] = self.df[column_name].apply(lambda tokens: ['<D>'] + tokens + ['</D>'])
    
    def _add_eos(self):
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df[column_name].apply(lambda arr: arr + ['<EOS>'])

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
                return remove_stopwords(tokens, level='moderate')
            return tokens
        
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df.apply(lambda row: filter_bengali_stopwords(row[column_name], row['language']), axis=1)

    def _remove_indonesian_stopwords(self):
        with open("tala-stopwords-indonesia.txt", "r") as f:
            stopword_list = [line.strip() for line in f]
        
        def filter_indonesian_stopwords(tokens, lang):
            if lang == "indonesian":
                return [token for token in tokens if token not in stopword_list]
            return tokens
        
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.df[column_name] = self.df.apply(lambda row: filter_indonesian_stopwords(row[column_name], row['language']), axis=1)

    def _count_all_tokens(self):
        for col in self.columns_to_tokenize:
            column_name = f'{col}_tokens'
            self.total_token_count = sum(self.df[column_name].apply(len))

