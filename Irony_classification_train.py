
# coding: utf-8

# In[ ]:


#import/install all packages at the top

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
from pathlib import Path


# In[ ]:


#import our own functions from feature_stats.py script
#each imported individually for explicity

from features_stats import get_all_tokens
from features_stats import get_words
from features_stats import get_punct
from features_stats import average_word_length
from features_stats import average_sent_length
from features_stats import check_sarcsymbol
from features_stats import count_uppercase
from features_stats import get_lemmas
from features_stats import get_punct_average
from features_stats import get_sentiment
from features_stats import get_indiv_punct
from features_stats import relative_count_wordtypes
from features_stats import get_entities


# In[ ]:


#insert your own file directory path here
file_directory = Path("/Users/laure/OneDrive/Desktop/")


# # Sections:
# 
# # (1) Import Dataset and Split
# - (1.1) Import dataset
# - (1.2) Split the data
# - (1.3) Check distribution (train, validation, test)
# - (1.4) Split train subset into "ironic" and "non-ironic" parts
# 
# # (2) Ironic Feature Extraction
# - (2.1) Average Word Count
# - (2.2) Average Sentence Count
# - (2.3) Punctuation Richness
# - (2.4) Sarcasm Symbol
# - (2.5) Upper-case Words
# - (2.6) Verb Lemmas
# - (2.7) Sentiment Classification
# 
# - (2.8) Individual Punctuation Count
# - (2.9) Word Type Count
# - (2.10) Named Entity Count
# 
# # (3) Non-ironic Feature Extraction
# - (3.1) Average Word Count
# - (3.2) Average Sentence Count
# - (3.3) Punctuation Richness
# - (3.4) Sarcasm Symbol
# - (3.5) Upper-case Words
# - (3.6) Verb Lemma
# - (3.7) Sentiment Classification
# 
# - (3.8) Individual Punctuation Count
# - (3.9) Part-of-Speech (POS) Count
# - (3.10) Named Entity Count
# 
# # (4) Get Final Summary Stats (ironic vs non-ironic)
# - (4.1) General Summary (save csv + visualisation)
# - (4.2) Individual Punctuation Count Summary (save csv + visualisation)
# - (4.3) Word Type Count Summary (save csv + visualisation)
# - (4.4) Named Entity Count Summary (save csv + visualisation)

# #  (1) Import Dataset and Split

# ### (1.1) Import the dataset

# In[ ]:


#import and read file in df with pandas (for better visualisation)
gold_label = pd.read_csv(file_directory / "irony-labeled.csv")

#if this doesn't run, use below:
#gold_label = pd.read_csv(file_directory / "irony-labeled.csv", engine = "python")


# In[ ]:


#rename the columns
gold_label.columns = ["Comment_Text", "Label"]


# In[ ]:


#checking for any missing values
missing_data = gold_label.isnull().sum()
missing_data


# In[ ]:


#counts number of each class 
gold_label["Label"].value_counts()

#1 ironic
#-1 non-ironic


# In[ ]:


print("This dataset of Ironic and Non-ironic Reddit Comments entails", len(gold_label), "items")


# ### (1.2) Split the data:
# #Train (70), Validation (10) and Test (20) sets
# #Scikit learn 'train_test_split' function twices gives the validation set

# In[ ]:


#Split to get two DFs (prep for split)
y = gold_label["Comment_Text"]
x = gold_label["Label"]


# In[ ]:


#Split the dataset into TEST and TRAIN sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

#Split the TRAIN set again to get VALIDATION set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=43)


# In[ ]:


#JOIN the series together to get final splits as DFs
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
val = pd.concat([X_val, y_val], axis=1)


# In[ ]:


# pandas '.shape' to see dataframe in form of tuples (no. of rows / cols)

print("Training Data Shape:", train.shape)
print("Testing Data Shape:", test.shape)
print("Validation Data Shape:", val.shape)


# ### (1.3) Check distribution
# #Summary of distribution of comments by labels (Non-ironic = -1, Ironic = 1)
# #Matplotlib to create THREE bar charts for visualisation for each of the sets:
#         (1) Train
#         (2) Validation
#         (3) Test

# In[ ]:


#TRAIN
fig = plt.figure(figsize=(8,6))
sns.barplot(x = train["Label"].unique(), 
            y=train["Label"].value_counts())

plt.title("Training Data: Ironic vs Non-ironic Reddit Comments")
plt.ylabel("Number of Reddit Comments")
plt.xlabel("Irony Labels (Not Ironic=-1; Ironic=1)")
plt.show()


# In[ ]:


#VALIDATION
fig = plt.figure(figsize=(8,6))
sns.barplot(x = val["Label"].unique(), 
            y=val["Label"].value_counts())

plt.title("Validation Data: Ironic vs Non-ironic Reddit Comments")
plt.ylabel("Number of Reddit Comments")
plt.xlabel("Irony Labels (Not Ironic=-1; Ironic=1)")
plt.show()


# In[ ]:


#TEST
fig = plt.figure(figsize=(8,6))
sns.barplot(x = test["Label"].unique(), 
            y=test["Label"].value_counts())

plt.title("Testing Data: Ironic vs Non-ironic Reddit Comments")
plt.ylabel("Number of Reddit Comments")
plt.xlabel("Irony Labels (Not Ironic=-1; Ironic=1)")
plt.show()


# ### (1.4) Split train subset into "ironic" and "non-ironic" parts

# In[ ]:


#check format of train df
train.head()


# In[ ]:


#Split Train set into "Ironic" and "Non-ironic" DataFrames
ironic_train = train[train["Label"] == 1]
nonironic_train = train[train["Label"] == -1]


# In[ ]:


#Convert into two dictionaries
ironic_dict = ironic_train.set_index(ironic_train.index).T.to_dict()
nonironic_dict = nonironic_train.set_index(nonironic_train.index).T.to_dict()

print(f"Training data contains {len(ironic_dict)} IRONIC comments")
print(f"Training data contains {len(nonironic_dict)} NON- IRONIC comments")


# # (2) Ironic Feature Extraction

# In[ ]:


#GET ALL TOKENS
ir_tokens = get_all_tokens(ironic_dict)


# In[ ]:


#Get list of ONLY words (no punct)
ir_word_list = get_words(ir_tokens)


# In[ ]:


#Get list of ONLY punct (no words)
ir_punct_list = get_punct(ir_tokens)


# In[ ]:


#Create df for total, full returns for irony
total_ir_train= pd.DataFrame({'Ironic Comment Parsed':ir_tokens})
total_ir_train["Tokens"] = ir_word_list
total_ir_train["Punctuation"] = ir_punct_list
total_ir_train.head()


# In[ ]:


#(2.1) AVERAGE WORD LENGTH
ir_average_word_leng = []
for comment in ir_word_list:
    ir_average_word_leng.append(average_word_length(comment))
    
#Create DataFrame for Summary of Irony STATS
summary_irony= pd.DataFrame({"Average Word Length": ir_average_word_leng})


# In[ ]:


#(2.2) AVERAGE SENTENCE LENGTH
ir_average_sent_leng = []
for x in ir_tokens:
    ir_average_sent_leng.append(average_sent_length(x))

#Add to Summary of Irony STATS df
summary_irony["Average Sentence Length"] = ir_average_sent_leng
summary_irony.head()


# In[ ]:


#(2.3) AVERAGE NUMBER OF SARCASM SYMBOL (/s)
ir_sarcfunc = []
for x in ir_tokens:
    ir_sarcfunc.append(check_sarcsymbol(x))

ir_sarcsymb_list = []        
for l in ir_sarcfunc:
    if len(l) >= 1:
        ir_sarcsymb_list.append(l)
    else:
        ir_sarcsymb_list.append([0])

#Remove list layer 
ir_sarcsymb_list = list(chain.from_iterable(ir_sarcsymb_list))

#Add result to Ironic Summary DF
summary_irony["Sarcasm Symbol (/s)"] = ir_sarcsymb_list


# In[ ]:


#(2.4) AVERAGE NUMBER OF UPPER CASE WORDS (total)
ir_uppercase_list = []
for b in ir_tokens:
    ir_uppercase_list.append((count_uppercase(b)))
    
#Remove list layer 
ir_uppercase_list = list(chain.from_iterable(ir_uppercase_list))

#Add result to Ironic Summary DF
summary_irony["Uppercase Average"] = ir_uppercase_list
summary_irony.head()


# In[ ]:


#(2.5) AVERAGE PUNCTUATION RICHNESS
ir_punct_avg = get_punct_average(ir_punct_list, ir_tokens)

#Add result to Ironic Summary DF
summary_irony["Punctuation Richness"] = ir_punct_avg
summary_irony.head()


# In[ ]:


#(2.6) AVERAGE NUMBER OF LEMMAS
ir_lemma_list = []
for doc in ir_tokens:
    ir_lemma_list.append(get_lemmas(doc))
    
len(ir_lemma_list)
    
summary_irony["Verb Lemma Average"] = ir_lemma_list
summary_irony.head()


# In[ ]:


#(2.7) SENTIMENT CLASSIFICATION
#1 = positive, -1 = negative

ir_sentiment = get_sentiment(ironic_dict)

summary_irony["Sentiment Classification"] = ir_sentiment 


# In[ ]:


#replace NAN values
summary_irony = summary_irony.replace(np.nan, 0)
# print(summary_irony)


# In[ ]:


#(2.8) AVERAGE FOR INDIVIDUAL PUNCTUATION MARKS 
ir_average_indivpunc_list = []
for x in ir_tokens:
    ir_average_indivpunc_list.append(get_indiv_punct(x))

#Create Summary DF for each individual Punctuation Mark
summary_irony_indivpunct = pd.DataFrame(ir_average_indivpunc_list)
summary_irony_indivpunct = summary_irony_indivpunct.replace(np.nan, 0)


# In[ ]:


#replace NAN values
summary_irony_indivpunct = summary_irony_indivpunct.replace(np.nan, 0)
summary_irony_indivpunct.head()


# In[ ]:


#(2.9) AVERAGE FOR ALL PARTS-OF-SPEECH (POS)
ir_average_pos_list = []
for comment in ir_tokens:
    ir_average_pos_list.append(relative_count_wordtypes(comment))

#Create Summary DF for POS
summary_irony_pos = pd.DataFrame(ir_average_pos_list)


# In[ ]:


#replace NAN values
summary_irony_pos = summary_irony_pos.replace(np.nan, 0)
summary_irony_pos.head()


# In[ ]:


#(2.10) AVERAGE FOR ALL NAMED ENTITIES 
ir_named_entity_list = []
for comment in ir_tokens:
    ir_named_entity_list.append(get_entities(comment))
    

#Create Summary DF for all Named Entities   
summary_irony_namedentity = pd.DataFrame(ir_named_entity_list)


# In[ ]:


#replace NAN values
summary_irony_namedentity = summary_irony_namedentity.replace(np.nan, 0)
summary_irony_namedentity.head()  


# # (3) Non-ironic Feature Extraction 

# In[ ]:


# GET ALL TOKENS
nonir_tokens = get_all_tokens(nonironic_dict)


# In[ ]:


# Get list of ONLY words (no punct)
nonir_word_list = get_words(nonir_tokens)


# In[ ]:


# Get list of ONLY punct (no words)
nonir_punct_list = get_punct(nonir_tokens)


# In[ ]:


#Create df for total, full returns for irony
total_nonir_train= pd.DataFrame({'non-ironic Comment Parsed':nonir_tokens})
total_nonir_train["Tokens"] = nonir_word_list
total_nonir_train["Punctuation"] = nonir_punct_list
total_nonir_train.head()


# In[ ]:


#(3.1) AVERAGE WORD LENGTH
nonir_average_word_leng = []
for comment in nonir_word_list:
    nonir_average_word_leng.append(average_word_length(comment))
    
#Create DataFrame for Summary of Irony STATS
summary_noirony= pd.DataFrame({"Average Word Length": nonir_average_word_leng})


# In[ ]:


#(3.2) AVERAGE SENTENCE LENGTH
nonir_average_sent_leng = []
for x in nonir_tokens:
    nonir_average_sent_leng.append(average_sent_length(x))

#Add to Summary of Irony STATS df
summary_noirony["Average Sentence Length"] = nonir_average_sent_leng
summary_noirony.head()


# In[ ]:


#(3.3) AVERAGE NUMBER OF SARCASM SYMBOL (/s)
nonir_sarcfunc = []
for x in nonir_tokens:
    nonir_sarcfunc.append(check_sarcsymbol(x))


nonir_sarcsymb_list = []        
for l in nonir_sarcfunc:
    if len(l) >= 1:
        nonir_sarcsymb_list.append(l)
    else:
        nonir_sarcsymb_list.append([0])

#Remove list layer 
nonir_sarcsymb_list = list(chain.from_iterable(nonir_sarcsymb_list))

#Add result to Ironic Summary DF
summary_noirony["Sarcasm Symbol (/s)"] = nonir_sarcsymb_list


# In[ ]:


#(3.4) AVERAGE NUMBER OF UPPER CASE WORDS (total)
nonir_uppercase_list = []
for b in nonir_tokens:
    nonir_uppercase_list.append((count_uppercase(b)))
    
    
#Remove list layer 
nonir_uppercase_list = list(chain.from_iterable(nonir_uppercase_list))

#Add result to Ironic Summary DF
summary_noirony["Uppercase Average"] = nonir_uppercase_list
summary_noirony.head()


# In[ ]:


#(3.5) AVERAGE PUNCTUATION RICHNESS
nonir_punct_avg = get_punct_average(nonir_punct_list, nonir_tokens)

#Add result to Ironic Summary DF
summary_noirony["Punctuation Richness"] = nonir_punct_avg
summary_noirony.head()


# In[ ]:


#(3.6) AVERAGE NUMBER OF LEMMAS
nonir_lemma_list = []
for doc in nonir_tokens:
    nonir_lemma_list.append(get_lemmas(doc))
    
summary_noirony["Verb Lemma Average"] = nonir_lemma_list
summary_noirony.head()


# In[ ]:


#(3.7) SENTIMENT CLASSIFICATION
#1 = positive, -1 = negative

nonir_sentiment = get_sentiment(nonironic_dict)

summary_noirony["Sentiment Classification"] = nonir_sentiment 


# In[ ]:


#replace NAN values

summary_noirony = summary_noirony.replace(np.nan)
summary_noirony.head()


# In[ ]:


#(3.8) AVERAGE FOR INDIVIDUAL PUNCTUATION MARKS 
nonir_average_indivpunc_list = []
for x in nonir_tokens:
    nonir_average_indivpunc_list.append(get_indiv_punct(x))

#Create Summary DF for each individual Punctuation Mark
summary_noirony_indivpunct = pd.DataFrame(nonir_average_indivpunc_list)


# In[ ]:


#replace NAN values

summary_noirony_indivpunct = summary_noirony_indivpunct.replace(np.nan, 0)
summary_noirony_indivpunct.head()


# In[ ]:


#(3.9) AVERAGE FOR ALL PARTS-OF-SPEECH (POS)
nonir_average_pos_list = []
for comment in nonir_tokens:
    nonir_average_pos_list.append(relative_count_wordtypes(comment))

#Create Summary DF for POS
summary_noirony_pos = pd.DataFrame(nonir_average_pos_list)


# In[ ]:


#replace NAN values
summary_noirony_pos = summary_noirony_pos.replace(np.nan, 0)
summary_noirony_pos.head()


# In[ ]:


#(3.10) AVERAGE FOR ALL NAMED ENTITIES  
nonir_named_entity_list = []
for comment in nonir_tokens:
    nonir_named_entity_list.append(get_entities(comment))
    

#Create Summary DF for all Named Entities   
summary_noirony_namedentity = pd.DataFrame(nonir_named_entity_list)


# In[ ]:


summary_noirony_namedentity = summary_noirony_namedentity.replace(np.nan, 0)
summary_noirony_namedentity.head()  


# # (4) Get Final Summary Stats (ironic vs non-ironic)
# 
# - (1) General Summary: create shared DF (1, -1) with mean, save to CSV and show visual
# - (2) Word Type summary: create shared DF (1, -1) with mean, save to CSV and show visual
# - (3) Punctuation Mark summary: create shared DF (1, -1) with mean, save to CSV and show visual
# - (4) Named Entity Recognition summary: create shared DF (1, -1) with mean, save to CSV and show visual

# ### (4.1) General Summary

# In[ ]:


#1) WORD LENGTH summary
ir_dist_word = summary_irony["Average Word Length"].mean()
non_ir_dist_word = summary_noirony["Average Word Length"].mean()


# In[ ]:


#2) SENTENCE LENGTH summary
ir_dist_sent = summary_irony["Average Sentence Length"].mean() 
non_ir_dist_sent = summary_noirony["Average Sentence Length"].mean()


# In[ ]:


#3) SARCASM SYMBOL (/s) summary
ir_sarcsymb = summary_irony["Sarcasm Symbol (/s)"].mean()
non_ir_sarcsymb = summary_noirony["Sarcasm Symbol (/s)"].mean()


# In[ ]:


#4) PUNCTUATION RICHNESS summary
ir_dist_punct = summary_irony["Punctuation Richness"].mean()
non_ir_dist_punct = summary_noirony["Punctuation Richness"].mean()


# In[ ]:


#5) UPPER-CASE LETTER summary
ir_uppercase = summary_irony["Uppercase Average"].mean()
non_ir_uppercase = summary_noirony["Uppercase Average"].mean()


# In[ ]:


#6) VERB LEMMA summary

ir_verblemma = summary_irony["Verb Lemma Average"].mean()
non_ir_verblemma = summary_noirony["Verb Lemma Average"].mean()


# In[ ]:


#7) SENTIMENT summary

ir_sentiment = summary_irony["Sentiment Classification"].mean()
nonir_sentiment = summary_noirony["Sentiment Classification"].mean()


# In[ ]:


#Create a Summary Stats df
# View both ironic/non-ironic average stats for each feature

summary_stats = pd.DataFrame(columns=["Average Word Length","Average Sentence Length"], index=["Ironic Comments", "Non-ironic Comments"])

summary_stats["Average Word Length"] = ir_dist_word, non_ir_dist_word
summary_stats["Average Sentence Length"] = ir_dist_sent, non_ir_dist_sent
summary_stats["Average '/s' symbol count"] = ir_sarcsymb, non_ir_sarcsymb
summary_stats["Average Upper-case Words"] = ir_uppercase, non_ir_uppercase
summary_stats["Punctuation Richness"] = ir_dist_punct, non_ir_dist_punct
summary_stats["Verb Lemma Average"] = ir_verblemma, non_ir_verblemma
summary_stats["Sentiment Classification"] = ir_sentiment, nonir_sentiment

summary_stats


# In[ ]:


#Save Master general table to CSV
summary_stats.to_csv(file_directory / "train_summary_general.csv")


# In[ ]:


#Create Visualisation for Summary-stats DataFrame
reset_indexbar = summary_stats.reset_index().melt(id_vars=["index"]) 

plt.figure(figsize=(12,9)) #size on-screen
sns.barplot(x="variable", y="value", hue="index", data=reset_indexbar, palette=["red", "lightsteelblue"] )
plt.xticks(rotation=45)


plt.title("Summary of statistics for Ironic and Non-ironic Reddit Comments: Training Set", fontsize=20)
plt.ylabel("Number", fontsize=20)
plt.xlabel("")
plt.tick_params(labelsize=15)

#save grouped bar chart as png file
plt.savefig("train_summary_general.png", bbox_inches = "tight")


# ### (4.2) POS Summary

# In[ ]:


#IRONIC POS
ir_pos = summary_irony_pos.mean()
ir_pos = pd.DataFrame(ir_pos) 

#Clean df (switch columns, rename index, move index inwards to right (for grouped bar later))
ir_pos = ir_pos.T #transpose columns w. rows


# In[ ]:


#NON-IRONIC POS
nonir_pos = summary_noirony_pos.mean()
nonir_pos = pd.DataFrame(nonir_pos) 

#clean df (switch columns, rename index, move index inwards to right (for grouped bar later))
nonir_pos = nonir_pos.T #transpose columns w. rows


# In[ ]:


#POS SUMMARY!
pos_frames = [ir_pos, nonir_pos]
summary_pos = pd.concat(pos_frames)
summary_pos.index = "Ironic", "Non-ironic"

summary_pos


# In[ ]:


#Save Master pos table to CSV

summary_pos.to_csv(file_directory / "train_summary_pos.csv")


# In[ ]:


#POS VISUAL SUMMARY
reset_indexbar_pos = summary_pos.reset_index().melt(id_vars=["index"]) 
#summary_stats.head()

plt.figure(figsize=(12,9)) #size on-screen
sns.barplot(x="variable", y="value", hue="index", data=reset_indexbar_pos, palette=["red", "lightsteelblue"] )
plt.xticks(rotation=45)

# #instead of palette, try -- color = "lightsteelblue"

plt.title("Relative Number for Parts of Speech (POS): Training Set", fontsize=20)
plt.ylabel("Number", fontsize=20)
plt.xlabel("POS (abbr.)", fontsize=20)

#plt.xlabel("Statistics Type", fontsize=15)
plt.tick_params(labelsize=15)

#save grouped bar chart as png file
plt.savefig("train_summary_POS.png", bbox_inches = "tight")


# ### (4.3) Punctuation Mark

# In[ ]:


#INDIVIDUAL PUNCT COUNT AVERAGE
#IRONIC
ir_punct_mean = summary_irony_indivpunct.mean()
ir_punct = pd.DataFrame(ir_punct_mean)
summary_ir_punct = ir_punct.T
summary_ir_punct.index.names = ["Ironic"]


# In[ ]:


#NON-IRONIC
nonir_punct_mean = summary_noirony_indivpunct.mean()
nonir_punct = pd.DataFrame(nonir_punct_mean)
summary_nonir_punct = nonir_punct.T
summary_nonir_punct.index.names = ["Non-ironic"]


# In[ ]:


indiv_punct_frames = [summary_ir_punct, summary_nonir_punct]
summary_punct_count = pd.concat(indiv_punct_frames, sort=True)
summary_punct_count.index = "Ironic", "Non-ironic"


# In[ ]:


#replace NaN with 0 for easier understanding
summary_punct_count = summary_punct_count.replace(np.nan, 0)


# In[ ]:


#Save Master punctuation table to CSV
summary_punct_count.to_csv(file_directory / "train_summary_puncttype.csv")


# In[ ]:


#INDIVIDUAL PUCNTUATION VISUAL SUMMARY
reset_indexbar_indpunct = summary_punct_count.reset_index().melt(id_vars=["index"]) 
#summary_punct_count.head()

plt.figure(figsize=(12,30)) #size on-screen
sns.barplot(x="value", y="variable", hue="index", data=reset_indexbar_indpunct, palette=["red", "lightsteelblue"] )
plt.xticks(rotation=45)

# #instead of palette, try -- color = "lightsteelblue"

plt.title("Average Number of Punctuation Marks: Training Set", fontsize=20)
plt.ylabel("Punctuation Mark", fontsize=20)
plt.xlabel("Number", fontsize=20)

#plt.xlabel("Statistics Type", fontsize=15)
plt.tick_params(labelsize=15)

#save grouped bar chart as png file
plt.savefig("train_summary_punctuation.png", bbox_inches = "tight")


# ### (4.4) NAMED-ENTITIES

# In[ ]:


#IRONIC
ir_entity = summary_irony_namedentity.mean()
ir_entity = pd.DataFrame(ir_entity) 

#clean df (switch columns, rename index, move index inwards to right (for grouped bar later))
ir_entity = ir_entity.T #transpose columns w. rows


# In[ ]:


#NON-IRONIC
non_ir_entity = summary_noirony_namedentity.mean()
non_ir_entity = pd.DataFrame(non_ir_entity)

# #clean df (switch columns, rename index, move index inwards to right (for grouped bar later))
non_ir_entity = non_ir_entity.T #transpose columns w. rows


# In[ ]:


#NAMED ENTITIES grouped summary df
entity_frames = [ir_entity, non_ir_entity]
summary_entity = pd.concat(entity_frames, sort=True)
summary_entity.index = "Ironic", "Non-ironic"
summary_entity.replace(np.nan, 0)


# In[ ]:


#Save Master general table to CSV
summary_entity.to_csv(file_directory / "train_summary_namedentity.csv")


# In[ ]:


#NER VISUAL SUMMARY


# In[ ]:


#NAMED ENTITY SUMMARY VISUAL 

reset_indexbar_entity = summary_entity.reset_index().melt(id_vars=["index"]) 

plt.figure(figsize=(12,9)) #size on-screen
sns.barplot(x="variable", y="value", hue="index", data=reset_indexbar_entity, palette=["red", "lightsteelblue"] )
plt.xticks(rotation=45)

plt.title("Average Number for Named Entity Recognition: Training Set", fontsize=20)
plt.ylabel("Number", fontsize=20)
plt.xlabel("Named Entity", fontsize=20)

plt.tick_params(labelsize=15)
#save grouped bar chart as png file
plt.savefig("train_summary_NER.png", bbox_inches = "tight")


# In[ ]:


#extra visual
#check for named entitiy labelling through spacy for IRONIC comments
#may have to scroll to very end as some comments with apparent no named entities
ir_ent_visual = spacy.displacy.render(ir_tokens, style="ent", jupyter=True)


# In[ ]:


#extra visual
#check for named entitiy labelling through spacy for NON-IRONIC
#may have to scroll to very end as some comments with apparent no named entities
nonir_ent_visual = spacy.displacy.render(nonir_tokens, style="ent", jupyter=True)

