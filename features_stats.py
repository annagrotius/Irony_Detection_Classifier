# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:37:10 2019

@author: Елизавета
"""

def get_all_tokens(test_dict):
    """Take dictionary and return list of comments as spacy docs"""
    comment_list = []
    for comment_index, label in test_dict.items():
        for key in label:
            text = label[key]
            if type(text) == str:
                comment_list.append(nlp(text))
    return comment_list

def get_words(listx):
    """Take a list (already parsed through SpaCy) remove punctuation and return list of word tokens"""
    ir_clean_docs = [] #remove punctuation

    for x in listx:
        clean_list = []
        for y in x:
            if y.pos_ != 'PUNCT':
                clean_list.append(y)
        ir_clean_docs.append(clean_list)
    return ir_clean_docs

def get_punct(listx):
    """Take a list (already parsed through spacy), remove words and return list of punctuation ONLY"""
    ir_punct = [] #only punctuation

    for x in listx:
        clean_list = []
        for y in x:
            if y.pos_ == 'PUNCT':
                clean_list.append(y)
        ir_punct.append(clean_list)
    return ir_punct

def average_word_length(doc):
    """Take doc and return average word length"""
    for token in doc:
        word = token.text
        average_word_length = sum(len(word) for word in doc) / len(doc)
    return(average_word_length)
    
def average_sent_length(doc):
    """Take doc and return average sentence length"""
    sent_list = []

    for sent in doc.sents:
        len_sent = len(sent)
        sent_list.append(len_sent)

    total = sum(sent_list)
    leng = len(sent_list)

    average_sent_length = total / leng
    return(average_sent_length)

def relative_count_wordtypes(doc):
    """Return relative count average for all word types i.e. nouns, pronouns, verbs etc with word type as key and average as value"""
    pos_tags = []
    for token in doc:
        pos_tags.append(token.pos_)
    counting = Counter(pos_tags) #returns dictionary with whole count for each word type in doc
    
    leng = len(doc) #overall length of doc (no. of tokens)
    new_dict = {}
    
    for key, value in counting.items(): #iterate over entire dict
        new_dict[key] = value/ leng
            
            
    return new_dict

def check_sarcsymbol(comment_list):
    """Take a list of comments (parsed through SpaCy); return list of items if "/s" is present [Reddit "/s" = sarcasm]"""
    sarcsymb = []
    for x in comment_list:
        for y in x:
            if y.text == "/s":
                sarcsymb.append(x)
    return(sarcsymb)
    
def get_punct_average(punctuation_list, token_comment_list):
    """Take preprocessed list of punctuation and full token list (MUST be of equal length); 
    Returns list of the average for ALL punctuation (based on number overall of tokens)
    for each comment""" 

    punct_count = []
    for comment in punctuation_list:
        punct_count.append(len(comment))

    len_comment = []
    for comment in token_comment_list:
        len_comment.append(len(comment))
    
    punct_count, len_comment = np.array(punct_count), np.array(len_comment) 
    averages = punct_count + len_comment/2
    return averages

def get_indiv_punct(doc):
    """Return relative count average for all word types i.e. nouns, pronouns, verbs etc with word type as key and average as value"""
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

def count_uppercase(doc):
    """Take nlp doc and return the average number of fully uppercase words for each comment as a list"""
    listd = []
    
    for token in doc:
        if token.is_upper == True:
            listd.append(token)
            
    counting = Counter(listd)
    
    my_dict = dict(counting)
    upper_count_avg = []
    
#     for key, value in my_dict.items():
    x = sum(my_dict.values())
    upper_count_avg.append(x)
#         if key == str:
#             my_dict[key] = sum(values)
    return upper_count_avg

def get_entities(doc):
    """Take nlp doc and return a dictionary with key as ent.labe_ and value as the average number"""
    entity = []
    for token in doc.ents:
        entity.append(token.label_)

    new_dict = Counter(entity)
    leng = len(doc)
    
    for key, value in new_dict.items():
        new_dict[key] = value / leng
        
    ent_dict = dict(new_dict)
    
    return ent_dict