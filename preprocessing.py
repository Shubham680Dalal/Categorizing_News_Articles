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
from bs4 import BeautifulSoup
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
import os
#os.system("python -m spacy download en_core_web_sm")
# import spacy
# #from spacy.cli import download
# nlp = spacy.load("en_core_web_sm")



from wordsegment import load, segment
from wordcloud import WordCloud

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
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

    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens."""
        return [word for word in tokens if word not in self.stop_words]

    def preprocess_dataframe(self, df, column="Article"):
        """Preprocess a dataframe column."""
        df = df.copy()
        df[column] = df[column].apply(self.clean_text)
        df["tokens"] = df[column].apply(self.tokenize_and_lemmatize)

        # Replace common/rare words and remove stopwords
        df["tokens"] = df["tokens"].apply(self.remove_stopwords)

        return df
    

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


def feature_revtransform(feature,scaler,numeric_cols,val):
    try:
        feature_index=numeric_cols.index(feature)
        feature_mean = scaler.mean_[feature_index]
        feature_std = scaler.scale_[feature_index]
        val=feature_mean+(val*feature_std)
        return val
    except:
        return val

## Model Interpretability
def extract_values_and_operators(st):
    # Match standalone numbers (including decimals and negatives) and comparison operators
    pattern = r"(?<!\w)-?\d+\.\d+|[<>=]+"
    matches = re.findall(pattern, st)
    # Convert numbers to float where applicable
    result = [float(m) if re.match(r"-?\d+\.\d+", m) else m for m in matches]
    return result

def update_lime_explanation(lime_exp, scaler, numeric_cols):
    """
    Update LIME explanation to use reverse-transformed values.
    """
    updated_explanation = []
    for condition, contrib in lime_exp.as_list():
        feature_name = re.search(r'([a-zA-Z_][a-zA-Z0-9_\.]*)\s*[<>=]', condition).group(1)  # Extract the feature name
        if feature_name in numeric_cols:
            # Extract numeric bounds and reverse-transform them

            bounds = extract_values_and_operators(condition)
            if len(bounds)==2:
                bounds[1]=feature_revtransform(feature_name,scaler,numeric_cols,bounds[1])
                updated_condition=feature_name+' '+str(bounds[0])+' '+str(bounds[1])
                updated_explanation.append((updated_condition, contrib))
            elif len(bounds)==4:
                bounds[0]=feature_revtransform(feature_name,scaler,numeric_cols,bounds[0])
                bounds[3]=feature_revtransform(feature_name,scaler,numeric_cols,bounds[3])
                updated_condition=str(bounds[0])+' '+str(bounds[1])+' '+feature_name+' '+str(bounds[2])+' '+str(bounds[3])
                updated_explanation.append((updated_condition, contrib))
            else:
                updated_explanation.append((condition, contrib))
        else:
            # Non-numeric features remain unchanged
            updated_explanation.append((condition, contrib))
    return updated_explanation

def show_lime_with_original_values(lime_exp, scaler, numeric_cols):

    updated_explanation = update_lime_explanation(lime_exp, scaler, numeric_cols)

    df_explanation = pd.DataFrame(updated_explanation, columns=["Feature Condition", "Contribution"])

    df_explanation["Absolute Contribution"] = df_explanation["Contribution"].abs()
    df_explanation = df_explanation.sort_values(by="Absolute Contribution", ascending=False)

    df_explanation = df_explanation.drop(columns=["Absolute Contribution"])
    return df_explanation