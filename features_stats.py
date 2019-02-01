# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:37:10 2019

@author: Lisa and Lauren
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
nlp = spacy.load("en_core_web_sm")
import numpy as np
from collections import Counter
from itertools import chain
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
tb = Blobber(analyzer=NaiveBayesAnalyzer())
import seaborn as sns
import matplotlib.pyplot as plt


def get_all_tokens(test_dict):
    """Take dictionary and return list of comments as SpaCy docs"""
    comment_list = []
    for comment_index, label in test_dict.items():
        for key in label:
            text = label[key]
            if type(text) == str:
                comment_list.append(nlp(text))
    return comment_list

def get_words(listx):
    """Take a list (already parsed through SpaCy), remove punctuation and return
    list of word tokens"""
    ir_clean_docs = [] #remove punctuation
    for x in listx:
        clean_list = []
        for y in x:
            if y.pos_ != 'PUNCT':
                clean_list.append(y)
        ir_clean_docs.append(clean_list)
    return ir_clean_docs

def get_punct(listx):
    """Take a list (already parsed through spacy), remove words and return list
    of punctuation ONLY"""
    ir_punct = [] #only punctuation
    for x in listx:
        clean_list = []
        for y in x:
            if y.pos_ == 'PUNCT':
                clean_list.append(y)
        ir_punct.append(clean_list)
    return ir_punct

def average_word_length(doc):
    """Take SpaCy doc and return average word length"""
    for token in doc:
        word = token.text
        average_word_length = sum(len(word) for word in doc) / len(doc)
    return(average_word_length)

def average_sent_length(doc):
    """Take SpaCy doc and return average sentence length"""
    sent_list = []
    for sent in doc.sents:
        len_sent = len(sent)
        sent_list.append(len_sent)
    total = sum(sent_list)
    leng = len(sent_list)
    average_sent_length = total / leng
    return(average_sent_length)

def check_sarcsymbol(doc):
    """Take SpaCy doc and return average number of "/s" symbols per comment
    [Reddit "/s" symbol = sarcasm]"""
    sarcsymb = []
    leng = len(doc)
    h = 1
    for x in doc:
        if x.text == "/s" or x.text == "/sarcasm" or x.text == "/sarc":
            sarcsymb.append(h/leng)
        else:
            pass
    return sarcsymb

def count_uppercase(doc):
    """Take SpaCy doc and return the average number of fully uppercase words
    within"""
    new_list = []
    leng = len(doc)
    for token in doc:
        if token.is_upper == True:
            new_list.append(token)
    counting = Counter(new_list)
    my_dict = dict(counting)
    upper_count_avg = []
    x = sum(my_dict.values())
    upper_count_avg.append(x/leng)
    return upper_count_avg

def get_lemmas(doc):
    """Take SpaCy doc and return average number of Verb Lemmas within"""
    lemma_list = [] #can't use set despite no duplicates - need to preserve order
    for token in doc:
        if token.pos_ == "VERB":
            if token.lemma_ not in lemma_list: #no duplicates
                lemma_list.append(token.lemma_)
        else:
            pass
    leng = len(doc)
    lemma_count = len(lemma_list)
    average_number_lemmas = lemma_count / leng
    return average_number_lemmas

def get_punct_average(punctuation_list, token_comment_list):
    """Take preprocessed SpaCy list of punctuation and preprocessed SpaCy list
    of all tokens (both same length); Returns numpy array with the average for
    ALL punctuation types (based on number overall of tokens) for each comment"""
    punct_count = []
    for comment in punctuation_list:
        punct_count.append(len(comment))
    len_comment = []
    for comment in token_comment_list:
        len_comment.append(len(comment))
    punct_count, len_comment = np.array(punct_count), np.array(len_comment)
    averages = punct_count + len_comment/2
    return averages

def get_sentiment(dicts):
    """Take dictionary of comments and return list of sentiment classifications
    (positive=1/negative=-1)"""
    comment_list = []
    for comment_index, label in dicts.items():
        for key in label:
            text = label[key]
            if type(text) == str:
                comment = tb(text)
                comment_list.append(comment.sentiment)
    classifications = []
    for sentiment in comment_list:
        classification, pos_score, neg_score = sentiment
        if classification == "pos":
            classifications.append(1)

        elif classification == "neg":
            classifications.append(-1)
        else:
            classifications.append("")
    return classifications

def get_indiv_punct(doc):
    """Take SpaCy doc and return average for each individual punctuation type
    in a dictionary; with punctuation type as key and average as value"""
    punc_tags = []
    for token in doc:
        if token.is_punct:
            punc_tags.append(token)
    #make each a string so not multiple keys with same vaues
    punc_tags = [str(punc) for punc in punc_tags]
    punc_tag_dict = Counter(punc_tags) #returns dictionary with whole count for each word type in doc
    leng = len(doc) #overall length of doc (no. of tokens)
    new_dict = {}
    for key, value in punc_tag_dict.items(): #iterate over entire dict
        new_dict[key] = value/ leng
    final_dict = dict(new_dict)
    return final_dict

def relative_count_wordtypes(doc):
    """Take SpaCy doc and return average for each individual Part-of-Speech (POS)
    in a dictionary; with POS as key and average as value"""
    pos_tags = []
    for token in doc:
        pos_tags.append(token.pos_)
    counting = Counter(pos_tags) #returns dictionary with whole count for each word type in doc
    leng = len(doc) #overall length of doc (no. of tokens)
    new_dict = {}
    for key, value in counting.items(): #iterate over entire dict
        new_dict[key] = value/ leng
    return new_dict

def get_entities(doc):
    """Take SpaCy doc and return average for each individual Named Entity in
    a dictionary; with entity label as key and average as value"""
    entity = []
    for token in doc.ents:
        entity.append(token.label_)
    new_dict = Counter(entity)
    leng = len(doc)
    for key, value in new_dict.items():
        new_dict[key] = value / leng
    ent_dict = dict(new_dict)
    return ent_dict
