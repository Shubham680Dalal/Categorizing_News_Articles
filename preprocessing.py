import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shap
import wordninja 

from scipy import stats
import itertools

import re
import string
import unicodedata
import pandas as pd
import contractions
from collections import Counter
from spellchecker import SpellChecker

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import spacy
from wordsegment import load, segment
from wordcloud import WordCloud


nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
load()

special_chars = r"!\"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~"
escaped_chars = re.escape(special_chars)

class TextPreprocessor:
    def __init__(self, common_threshold=1000, rare_threshold=1):
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()
        self.stop_words = set(stopwords.words("english"))
        self.vocab = Counter()
        self.common_words = set()
        self.rare_words = set()
        self.common_threshold = common_threshold  # absolute count
        self.rare_threshold = rare_threshold  # Absolute count

    def clean_text(self, text):
        """Preprocess text - remove HTML, punctuations, special characters, etc."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
        text = contractions.fix(text)  # Expand contractions
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # Remove mentions
        text = re.sub(r"#\S+", "", text)  # Remove hashtags
        text = re.sub(r"\bRT\b", "", text)  # Remove RT
        text = re.sub(r"\S+@\S+", "", text)  # Remove emails
        text = re.sub(r"[0-9]+", "", text)  # Remove numeric vals
        text = self.remove_dots(text)
        text = re.sub(rf"[{escaped_chars}]", "", text)  # Remove punctuation
        text = ''.join(c for c in text if not unicodedata.combining(c))  # Remove accented chars
        text = re.sub(r"\s+", " ", text).strip()  # Remove multiple spaces
        
        tokens = word_tokenize(text)
        tokens = [wordninja.split(word) for word in tokens]   ## breaks any concatenated words like "lookabout" to "look" and "about"
        tokens = [subword for word in tokens for subword in word] 
        
        pos_tags = pos_tag(tokens)
        non_nouns=[]
        for word,tag in pos_tags:     ## only spellcheck non noun words
            if not tag.startswith("NN"):
                non_nouns.append(word)
            
        tokens = [self.spell.correction(word) if (word not in self.spell.known([word])) and\
                  (word in non_nouns) and (self.spell.correction(word) is not None) else word for word in tokens]
        
        return " ".join(tokens)
    
    
    def remove_dots(self, text):
        return re.sub(r'\b([A-Z])(?:\.(?=[A-Z]))+\b', lambda m: m.group(0).replace(".", ""), text)
    def tokenize_and_lemmatize(self, text):
        """Tokenize, spell correct, and lemmatize text."""
        tokens = word_tokenize(text)
        
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def build_vocab(self, texts):
        """Build vocabulary from a list of tokenized texts."""
        for tokens in texts:
            self.vocab.update(tokens)

        # Identify common words (occurring in more than common_threshold)
        total_docs = len(texts)
        self.common_words = {word for word, count in self.vocab.items() if count > self.common_threshold}

        # Identify rare words (appearing less than rare_threshold times)
        self.rare_words = {word for word, count in self.vocab.items() if count < self.rare_threshold}

    def replace_common_rare(self, tokens):
        """Replace common words with <COMMON> and rare words with <UNK>."""
        return [word for word in tokens if word not in self.rare_words and word not in self.common_words]

    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens."""
        return [word for word in tokens if word not in self.stop_words]

    def preprocess_dataframe(self, df, column="Article"):
        """Preprocess a dataframe column."""
        df = df.copy()
        df[column] = df[column].apply(self.clean_text)
        df["tokens"] = df[column].apply(self.tokenize_and_lemmatize)

        # Build vocab after tokenization
        self.build_vocab(df["tokens"].tolist())

        # Replace common/rare words and remove stopwords
        df["tokens"] = df["tokens"].apply(self.replace_common_rare)
        df["tokens"] = df["tokens"].apply(self.remove_stopwords)

        return 
    

def filter_numeric(data,coll):
    enum_coll=list(enumerate(coll))
    cols=data.select_dtypes(include=['number']).columns
    cc=[]
    for col in cols:
        if (data[col].nunique(dropna=True)>2) or\
        ((sorted(list(data[col].dropna().unique()))!=[0,1]) and\
        (sorted(list(data[col].dropna().unique()))!=[0]) and\
        (sorted(list(data[col].dropna().unique()))!=[1])):
            cc.append(col)

    cols=cc
    numeric_cols=list(set(cols).intersection(set(coll)))
    li=[sett[1] for sett in enum_coll if sett[1] in numeric_cols]
    return li

def standardize_data(d,cols,strategy='standard'):
    if strategy=='standard':
        scalar=StandardScaler()
    else:
        scalar=MinMaxScaler()
    data=d.copy()
    cols=filter_numeric(data,cols)
    #print(cols)
    data.loc[:,cols]=pd.DataFrame(scalar.fit_transform(data[cols]),columns=cols,index=data.index)

    return data,scalar,cols

def transform_standardize_data(d,cols,scalar):
    data=d.copy()
    data.loc[:,cols]=pd.DataFrame(scalar.transform(data[cols]),columns=cols,index=data.index)
    return data

def revtransform_standardize_data(d,cols,scalar):
    data=d.copy()
    data.loc[:,cols]=pd.DataFrame(scalar.inverse_transform(data[cols]),columns=cols,index=data.index)
    return data

def transform_test(test_data, col, vectorizer):
    dd=test_data.copy()
    dd[col] = dd[col].apply(lambda x: " ".join(x))
    test_vectors = vectorizer.transform(dd[col])
    return pd.DataFrame(test_vectors.toarray(),columns=vectorizer.get_feature_names_out())