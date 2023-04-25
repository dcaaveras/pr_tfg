# functions for text preprocessing

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import pattern
from pattern.en import lemma, lexeme
import contractions
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
import nltk
from collections import defaultdict
import gensim
import regex
import nltk
import spacy


# functions for text preprocessing
stop_words = set(stopwords.words("english"))
def remove_Stopwords(text): 
    #stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    
    sentence = [w for w in words if not w in stop_words]
    return " ".join(sentence)


def clean_text(text): 
    # remove punctuation
    delete_dict = {sp_character:"" for sp_character in string.punctuation}
    delete_dict[" "] = " "
    table=str.maketrans(delete_dict)
    text1=text.translate(table)
    
    textArr = text1.split()
    text2= " ".join([w for w in textArr])
    
    # lower case + remove \n and ' that has not been removed in the previous function
    return text2.lower().replace("'", "").replace("\n", "")



def c2(text): 
    return text.replace("U.S.", "USA ").replace("US", "USA ").replace("US.","USA ").replace("United States", "USA")



def full_preprocess(ctext): 
    text_processed = regex.sub('^((http|https)://)[-a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)$', '', ctext) # eliminar enlaces   
    text_processed = regex.sub(r'www\S+', '', text_processed) # remove links startign with www
    text_processed = regex.sub('([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+', '', text_processed) # eliminar correos electronicos
    text_processed = regex.sub('/(^|[^@\w])@(\w{1,15})\b/', '', text_processed) # eliminar menciones de usuario
    text_processed = c2(text_processed)
    text_processed = contractions.fix(text_processed)# expand contractions
    text_processed = clean_text(text_processed) # remove punctuation, lower case, ...
    text_processed = regex.sub(r'\w*\d\w*','', text_processed)# remove words that contains a digit (eg: name57)
    text_processed = remove_Stopwords(text_processed)#remove stopwords
    text_processed = regex.sub(r'[0-9]+','', text_processed)# remove digits
    # text_processed = lemmatizer_func(text_processed)

    text_split = text_processed.split()
    if len(text_split) > 0:
        # print(text_split)
        text_processed=[palabra for palabra in text_split if len(palabra) > 2] # remove words that have 2 or fewer characters
        text_processed = " ".join(text_processed)
        #text_processed = " ".join(list(map(lambda x: lemma(x.strip()), text_split)))
    else: 
        return ""

    #bigrams and trigrams
    """text_split = text_processed.split()
    bigrams = ngrams(text_split, 2)
    trigrams = ngrams(text_split, 3)
    text_processed = text_processed + " " + " ".join(["_".join(bigram) for bigram in bigrams]) + " " + " ".join(["_".join(trigram) for trigram in trigrams])"""

    return text_processed



"""
LEMMATIZATION
POS TAGGING
BIGRAMS Y TRIGRAMS SIGUIENDO LO SIGUIENTE: 
    adjective-noun (ADJ NOUN)
    noun-noun
    adjective-adjective-noun
    adjective-noun-noun
    noun-adjective-noun
    noun-noun-noun
    noun-preposition-noun (NOUN - ADP - NOUN)
FILTRAR POR 1% Y 99% DURANTE PROCESO tf idf  
"""

sp = spacy.load('en_core_web_sm')
def final(text): 
    final_text = []
    doc = sp(text)
    
    for i in range(len(doc)): 
        # get NOUNS
        if doc[i].pos_ == "NOUN" or doc[i].pos_ =="PROPN": 
            final_text.append(doc[i].lemma_)
            
            try: 
                if doc[i+1].pos_=="NOUN" or doc[i+1].pos_ =="PROPN": 
                    final_text.append(doc[i].lemma_ + "_"+doc[i+1].lemma_)
                    if doc[i+2].pos_=="NOUN" or doc[i+2].pos_ =="PROPN": 
                        final_text.append(doc[i].lemma_ + "_"+doc[i+1].lemma_ +"_" +doc[i+2].lemma_)  
                     
                if doc[i+1].pos_=="ADJ" and (doc[i+2].pos_=="NOUN" or doc[i+2].pos_ =="PROPN"): 
                    final_text.append(doc[i].lemma_ + "_"+doc[i+1].lemma_ +"_" +doc[i+2].lemma_)  
            except Exception as e:
                pass
            
            
        try: 
            if doc[i].pos_=="ADJ": 
                if doc[i+1].pos_=="NOUN" or doc[i+2].pos_ =="PROPN": 
                    final_text.append(doc[i].lemma_ + "_"+doc[i+1].lemma_)
                    if doc[i+2].pos_=="NOUN" or doc[i+2].pos_ =="PROPN": 
                        final_text.append(doc[i].lemma_ + "_"+doc[i+1].lemma_ +"_" +doc[i+2].lemma_)
                elif doc[i+1].pos_=="ADJ" and (doc[i+2].pos_=="NOUN" or doc[i+2].pos_ =="PROPN"): 
                    final_text.append(doc[i].lemma_ + "_"+doc[i+1].lemma_ +"_" +doc[i+2].lemma_)

            
            if doc[i+1].pos_=="ADP" and (doc[i+2].pos_=="NOUN" or doc[i+2].pos_ =="PROPN"): 
                final_text.append(doc[i].lemma_ + "_"+doc[i+1].lemma_ +"_" +doc[i+2].lemma_)  
               
        except Exception as e:
            pass
    return " ".join(final_text)