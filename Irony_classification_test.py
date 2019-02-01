
# coding: utf-8

# In[ ]:


#import/install all packages at the top

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
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

from classification_stats import get_classification_ironic
from classification_stats import get_classification_non_ironic
from classification_stats import final_predicition_results
from classification_stats import accuracy


# In[ ]:


#insert your own file directory path here
file_directory = Path("/Users/laure/OneDrive/Desktop/")


# # Sections:
# 
# # (1) Import Dataset and Split
# - (1.1) Import dataset
# - (1.2) Split the data
# - (1.3) Check distribution (validation)
# 
# 
# # (2) Feature Extraction (with summary tables)
# - (2.1) Average Word Count
# - (2.2) Average Sentence Count
# - (2.3) Punctuation Richness
# - (2.4) Sarcasm Symbol
# - (2.5) Upper-case Words
# - (2.6) (Verb) Lemmas
# - (2.7) Sentiment Classification
# 
# - (2.8) Individual Punctuation Count
# - (2.9) Word Type Count
# - (2.10) Named Entity Count
# 
# 
# # (3) Classification
# - (3.1) feature weight for each linguistic feature within each comment (summary_general) 
# - (3.2) feature weight for each linguistic feature within each comment (summary_pos)
# - (3.3) feature weight for each linguistic feature within each comment (summary_punct)
# - (3.4) feature weight for each linguistic feature within each comment (summary_ner)
# 
# 
# # (4) Classification Results
# - (4.1) Final feature weight calculation for each comment against ironic averages
# - (4.2) Final feature weight calculation for each comment against non-ironic averages
# - (4.3) Final predictor label generation (assign predictor labels to each comment)
# 
# 
# # (5) Accuracy Score
# - (5.1) Calculate number of correctly predicted labels

# # (1) Import and Split

# ### (1.1) Import the dataset:

# In[ ]:


#Import and Read file as DF with PANDAS (for better visualisation)
gold_label = pd.read_csv(file_directory / "irony-labeled.csv")

#if this doesn't run, use below:
#gold_label = pd.read_csv(file_directory / "irony-labeled.csv", engine = "python")


# In[ ]:


#Rename the columns
gold_label.columns = ["Comment_Text", "Label"]


# ### (1.2) Split the data: 

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


# ### Check the distributions (test)

# In[ ]:


#TEST
fig = plt.figure(figsize=(8,6))
sns.barplot(x = test["Label"].unique(), 
            y=test["Label"].value_counts())

plt.title("Testing Data: Ironic vs Non-ironic Reddit Comments")
plt.ylabel("Number of Reddit Comments")
plt.xlabel("Irony Labels (Not Ironic=-1; Ironic=1)")
plt.show()


# In[ ]:


#Check number of comments labelled as ironic vs non-ironic
ironic_test = test[test["Label"] == 1]
nonironic_test = test[test["Label"] == -1]

print(f"Testing data contains {len(ironic_test)} IRONIC comments")
print(f"Testing data contains {len(nonironic_test)} NON- IRONIC comments")


# In[ ]:


#Convert TEST set into a dictionary
test_dict = test.set_index(test.index).T.to_dict()

print(len(test_dict))


# # (2) Feature Extraction (with summary tables)

# In[ ]:


#GET ALL TOKENS
tokens = get_all_tokens(test_dict)


# In[ ]:


#Get list of ONLY words (no punct)
word_list = get_words(tokens)


# In[ ]:


#Get list of ONLY punct (no words)
punct_list = get_punct(tokens)


# In[ ]:


#(2.1) AVERAGE WORD LENGTH
average_word_list = []
for comment in word_list:
    average_word_list.append(average_word_length(comment))

print(len(average_word_list))    
    
#Create DataFrame for Summary of Irony STATS
summary= pd.DataFrame({"Average Word Length": average_word_list})


# In[ ]:


#Create df for total, full returns for irony
total_test= pd.DataFrame({'Comment Parsed':tokens})
total_test["Tokens"] = word_list
total_test["Punctuation"] = punct_list
total_test.head()


# In[ ]:


#(2.2) AVERAGE SENTENCE LENGTH
average_sentence_list = []
for x in tokens:
    average_sentence_list.append(average_sent_length(x))

#Add to Summary of Irony STATS df
summary["Average Sentence Length"] = average_sentence_list
summary.head()


# In[ ]:


#(2.3) AVERAGE NUMBER OF SARCASM SYMBOL (/s)
sarcfunc = []
for x in tokens:
    sarcfunc.append(check_sarcsymbol(x))


sarcsymb_list = []        
for l in sarcfunc:
    if len(l) >= 1:
        sarcsymb_list.append(l)
    else:
        sarcsymb_list.append([0])

#Remove list layer 
sarcsymb_list = list(chain.from_iterable(sarcsymb_list))



summary["Average '/s' symbol count"] = sarcsymb_list


# In[ ]:


#(2.4) AVERAGE NUMBER OF UPPER CASE WORDS (total)
uppercase_list = []
for b in tokens:
    uppercase_list.append((count_uppercase(b)))
    
#Remove list layer 
uppercase_list = list(chain.from_iterable(uppercase_list))

summary["Average Upper-case Words"] = uppercase_list
summary.head()


# In[ ]:


#(2.5) AVERAGE PUNCTUATION RICHNESS
average_punct_list = get_punct_average(punct_list, tokens)

summary["Punctuation Richness"] = average_punct_list
summary.head()


# In[ ]:


#(2.6) AVERAGE NUMBER OF LEMMAS
lemma_list = []
for doc in tokens:
    lemma_list.append(get_lemmas(doc))
    
summary["Verb Lemma Average"] = lemma_list
summary.head()


# In[ ]:


#(2.7) SENTIMENT CLASSIFICATION
#1 = positive, -1 = negative
sentiment = get_sentiment(test_dict)

summary["Sentiment Classification"] = sentiment 


# In[ ]:


#replace NAN values
summary = summary.replace(np.nan, 0)
summary.head()


# In[ ]:


#Save test general table to CSV
summary.to_csv(file_directory / "test_summary_general.csv")


# In[ ]:


#(2.8) AVERAGE FOR INDIVIDUAL PUNCTUATION MARKS 
average_indiv_punc_list = []
for x in tokens:
    average_indiv_punc_list.append(get_indiv_punct(x))


summary_indiv_punct = pd.DataFrame(average_indiv_punc_list)


# In[ ]:


#replace NAN values
summary_indiv_punct = summary_indiv_punct.replace(np.nan, 0)
pd.options.display.max_columns = 40
summary_indiv_punct.head()


# In[ ]:


#Save test punctuation table to CSV
summary_indiv_punct.to_csv(file_directory / "test_summary_puncttype.csv")


# In[ ]:


#(2.9) AVERAGE FOR ALL PARTS-OF-SPEECH (POS)
average_pos_list = []
for comment in tokens:
    average_pos_list.append(relative_count_wordtypes(comment))

#Create Summary DF for POS
summary_posdf = pd.DataFrame(average_pos_list)


# In[ ]:


#replace NAN values
summary_posdf = summary_posdf.replace(np.nan, 0)
summary_posdf.head()


# In[ ]:


#Save test pos table to CSV
summary_wordtypedf.to_csv(file_directory / "test_summary_pos.csv")


# In[ ]:


#(2.10) AVERAGE FOR ALL NAMED ENTITIES 
named_entity_list = []
for comment in tokens:
    named_entity_list.append(get_entities(comment))
    
summary_named_entity = pd.DataFrame(named_entity_list)


# In[ ]:


#replace NAN values
summary_named_entity = summary_named_entity.replace(np.nan, 0) 
summary_named_entity.head()


# In[ ]:


#Save test general table to CSV
summary_named_entity.to_csv(file_directory / "test_summary_namedentity.csv")


# # (5) Classification
# 
# Steps:
# - (1) Import Master DF ###(1) GENERAL
# - (2) Get Results for each comparison using classification function (1) Ir, (2) Non-ir
# - (3) Create PredictorDF for (1) Ironic, (2) Non-ironic

# ''''''''''''''''''''''''''''''
# - (4) Import Master DF ###(2) POS
# - (5) Get Results for each comparison using classification function (1) Ir, (2) Non-ir
# - (6) Add to each PredictorDF for (1) Ironic, (2) Non-ironic

# ''''''''''''''''''''''''''''''
# - (7) Import Master DF ###(3) NAMED ENTITY
# - Repeat steps 5 & 6

# ''''''''''''''''''''''''''''''
# - (8) Import Master DF ###(4) PUNCTUATION
# - Repeat steps 5 & 6 

# ### (3.1) summary_general 

# In[ ]:


#import GENERAL summary table
mastergeneral_df = pd.read_csv(file_directory / "train_summary_general.csv")
mastergeneral_df.head()

mastergeneral_df = mastergeneral_df.rename(columns={mastergeneral_df.columns[0]: "Class"})

mastergeneral_df


# In[ ]:


#get results from classification function for IRONIC
ironic_average_word_length = get_classification_ironic(mastergeneral_df, summary, 1, 0, 15)
ironic_average_sent_length = get_classification_ironic(mastergeneral_df, summary, 2, 1, 1)
ironic_average_sarcsymb = get_classification_ironic(mastergeneral_df, summary, 3, 2, 1)
ironic_average_uppercase = get_classification_ironic(mastergeneral_df, summary, 4, 3, 1)
ironic_punct_richness = get_classification_ironic(mastergeneral_df, summary, 5, 4, 100)
ironic_average_verblemma = get_classification_ironic(mastergeneral_df, summary, 6, 5, 1)
ironic_average_sentiment = get_classification_ironic(mastergeneral_df, summary, 7, 6, 1)


# In[ ]:


#IRONIC
#Create PREDICTOR DATAFRAME with classifications (all features)

ironic_predictor_df = pd.DataFrame(ironic_average_word_length)
ironic_predictor_df.columns = ['WORD LENGTH'] + ironic_predictor_df.columns.tolist()[1:]

ironic_predictor_df["SENTENCE LENGTH"] = ironic_average_sent_length
ironic_predictor_df["PUNCT RICH"] = ironic_punct_richness
ironic_predictor_df["SARC SYMB /S"] = ironic_average_sarcsymb
ironic_predictor_df["UPPERCASE"] = ironic_average_uppercase
ironic_predictor_df["Verb Lemma Average"] = ironic_average_verblemma
ironic_predictor_df["Sentiment Classification"] = ironic_average_sentiment

ironic_predictor_df.head()


# In[ ]:


#get results from classification function for NON-IRONIC
non_ironic_average_word_length = get_classification_non_ironic(mastergeneral_df, summary, 1, 0, 15)
non_ironic_average_sent_length= get_classification_non_ironic(mastergeneral_df, summary, 2, 1, 1)
non_ironic_average_sarcsymb= get_classification_non_ironic(mastergeneral_df, summary, 3, 2, 1)
non_ironic_average_uppercase = get_classification_non_ironic(mastergeneral_df, summary, 4, 4, 1)
non_ironic_punct_richness = get_classification_non_ironic(mastergeneral_df, summary, 5, 3, 100)
non_ironic_average_verblemma = get_classification_non_ironic(mastergeneral_df, summary, 6, 5, 1)
non_ironic_average_sentiment = get_classification_non_ironic(mastergeneral_df, summary, 7, 6, 1)


# In[ ]:


#NON-IRONIC
#Create PREDICTOR DATAFRAME with classifications (all features)

non_ironic_predictor_df = pd.DataFrame(non_ironic_average_word_length)
non_ironic_predictor_df.columns = ['WORD LENGTH'] + non_ironic_predictor_df.columns.tolist()[1:]

non_ironic_predictor_df["SENTENCE LENGTH"] = non_ironic_average_sent_length
non_ironic_predictor_df["SARC SYMB /S"] = non_ironic_average_sarcsymb
non_ironic_predictor_df["PUNCT RICH"] = non_ironic_punct_richness
non_ironic_predictor_df["UPPERCASE"] = non_ironic_average_uppercase
non_ironic_predictor_df["Verb Lemma Average"] = non_ironic_average_verblemma
non_ironic_predictor_df["Sentiment Classification"] = non_ironic_average_sentiment
non_ironic_predictor_df.head()


# ### (3.2) summary_POS 

# In[ ]:


#import POS summary table

masterpos_df = pd.read_csv(file_directory / "train_summary_pos.csv")
masterpos_df.head()

masterpos_df = masterpos_df.rename(columns={mastergeneral_df.columns[0]: "Class"}) 
masterpos_df.head()


# In[ ]:


#get results from classification function for IRONIC
#E.g. PRON, PROPN, NOUN

# ironic_PRON_dist= get_classification_ironic(masterpos_df, summary_posdf, 10, 9, 1)
# ironic_PROPN_dist_length= get_classification_ironic(masterpos_df, summary_posdf, 11, 10, 1)
ironic_NOUN_dist = get_classification_ironic(masterpos_df, summary_posdf, 7, 6, 1)
ironic_CCONJ_dist = get_classification_ironic(masterpos_df, summary_posdf, 4, 3, 1)
ironic_VERB_dist = get_classification_ironic(masterpos_df, summary_posdf, 15, 14, 1)


# In[ ]:


#Add to IRONIC PREDICTOR DATAFRAME
# ironic_predictor_df["PRON"] = ironic_PRON_dist
# ironic_predictor_df["PROPN"] = ironic_PROPN_dist_length
ironic_predictor_df["NOUN"] = ironic_NOUN_dist
ironic_predictor_df["CCONJ"] = ironic_CCONJ_dist
ironic_predictor_df["VERB"] = ironic_VERB_dist
ironic_predictor_df = ironic_predictor_df.replace(np.nan, 0)
ironic_predictor_df.head()


# In[ ]:


#get results from classification function for NON-IRONIC
#E.g. PRON, PROPN, NOUN

# nonironic_PRON_dist= get_classification_non_ironic(masterpos_df, summary_posdf, 10, 9, 1)
# nonironic_PROPN_dist_length= get_classification_non_ironic(masterpos_df, summary_posdf, 11, 10, 1)
nonironic_NOUN_dist = get_classification_non_ironic(masterpos_df, summary_posdf, 7, 6, 1)
nonironic_CCONJ_dist = get_classification_non_ironic(masterpos_df, summary_posdf, 4, 3, 1)
nonironic_VERB_dist = get_classification_non_ironic(masterpos_df, summary_posdf, 15, 14, 1)


# In[ ]:


#Add to NON-IRONIC PREDICTOR DATAFRAME
# non_ironic_predictor_df["PRON"] = nonironic_PRON_dist
# non_ironic_predictor_df["PROPN"] = nonironic_PROPN_dist_length
non_ironic_predictor_df["NOUN"] = nonironic_NOUN_dist
non_ironic_predictor_df["CCONJ"] = nonironic_CCONJ_dist
non_ironic_predictor_df["VERB"] = nonironic_VERB_dist
non_ironic_predictor_df.head()


# ### (3.3) summary_NER 

# In[ ]:


#import NER summary table
masterentity_df = pd.read_csv(file_directory / "train_summary_namedentity.csv")
masterentity_df.head()

masterentity_df.rename(columns={mastergeneral_df.columns[0]: "Class"})
masterentity_df = masterentity_df.replace(np.nan, 0)
masterentity_df


# In[ ]:


#get results from classification function for IRONIC
#E.g. PERSON, LOC, GPE, LANGUAGE (none)

ironic_PERSON_dist= get_classification_ironic(masterentity_df, summary_named_entity, 14, 13, 1)
ironic_LOC_dist_length= get_classification_ironic(masterentity_df, summary_named_entity, 8, 7, 1)
ironic_GPE_dist = get_classification_ironic(masterentity_df, summary_named_entity, 5, 4, 30)
ironic_LANGUAGE_dist = get_classification_ironic(masterentity_df, summary_named_entity, 6, 5, 1)
ironic_ORG_dist= get_classification_ironic(masterentity_df, summary_named_entity, 12, 11, 1)


# In[ ]:


#Add to IRONIC PREDICTOR DATAFRAME
ironic_predictor_df["PERSON"] = ironic_PERSON_dist
ironic_predictor_df["LOC"] = ironic_LOC_dist_length
ironic_predictor_df["GPE"] = ironic_GPE_dist
ironic_predictor_df["LANGUAGE"] = ironic_LANGUAGE_dist
ironic_predictor_df["ORG"] = ironic_ORG_dist

ironic_predictor_df.head()


# In[ ]:


#get results from classification function for NON- IRONIC
#E.g. PERSON, LOC, GPE, LANGUAGE (none)

nonironic_PERSON_dist= get_classification_non_ironic(masterentity_df, summary_named_entity, 14, 13, 1)
nonironic_LOC_dist= get_classification_non_ironic(masterentity_df, summary_named_entity, 8, 7, 1)
nonironic_GPE_dist = get_classification_non_ironic(masterentity_df, summary_named_entity, 5, 4, 30)
nonironic_LANGUAGE_dist = get_classification_non_ironic(masterentity_df, summary_named_entity, 6, 5, 1)
nonironic_ORG_dist = get_classification_non_ironic(masterentity_df, summary_named_entity, 12, 11, 1)


# In[ ]:


#Add to NON-IRONIC PREDICTOR DATAFRAME
non_ironic_predictor_df["PERSON"] = nonironic_PERSON_dist
non_ironic_predictor_df["LOC"] = nonironic_LOC_dist
non_ironic_predictor_df["GPE"] = nonironic_GPE_dist
non_ironic_predictor_df["LANGUAGE"] = nonironic_LANGUAGE_dist
non_ironic_predictor_df["ORG"] = nonironic_ORG_dist


non_ironic_predictor_df = non_ironic_predictor_df.replace(np.nan, 0)
non_ironic_predictor_df.head()


# ### (3.4) summary_punct 

# In[ ]:


#import PUNCTUATION summary table
masterpunct_df = pd.read_csv(file_directory / "train_summary_puncttype.csv")
masterpunct_df.head()

masterpunct_df = masterpunct_df.rename(columns={mastergeneral_df.columns[0]: "Class"})
# len(masterpunct_df.columns)
pd.options.display.max_columns = 40
masterpunct_df


# In[ ]:


#get results from classification function for IRONIC
#E.g. !, ', *, :( 

ironic_exclam_dist = get_classification_ironic(masterpunct_df, summary_indiv_punct, 1, 0, 10)
ironic_apost_dist_length = get_classification_ironic(masterpunct_df, summary_indiv_punct, 7, 6, 1)
ironic_hash_dist_length = get_classification_ironic(masterpunct_df, summary_indiv_punct,4, 3, 8)
# ironic_leftsquare_dist_length = get_classification_ironic(masterpunct_df, summary_indiv_punct, 30, 29, 8)
ironic_star_dist = get_classification_ironic(masterpunct_df, summary_indiv_punct, 10, 9, 1)
# ironic_quest_dist = get_classification_ironic(masterpunct_df, summary_indiv_punct, 29, 28, 1)
ironic_sademoji_dist = get_classification_ironic(masterpunct_df, summary_indiv_punct, 25, 24, 1)


# In[ ]:


# Add to IRONIC PREDICTOR DATAFRAME
ironic_predictor_df["!"] = ironic_exclam_dist
ironic_predictor_df["'"] = ironic_apost_dist_length
ironic_predictor_df["#"] = ironic_hash_dist_length
ironic_predictor_df["*"] = ironic_star_dist
ironic_predictor_df[":("] = ironic_sademoji_dist
# ironic_predictor_df["["] = ironic_leftsquare_dist_length

ironic_predictor_df.head()


# In[ ]:


#get results from classification function for NON- IRONIC
#E.g. !, ', *, :( 

nonironic_exclam_dist= get_classification_non_ironic(masterpunct_df, summary_indiv_punct, 1, 0, 10)
nonironic_apost_dist_length= get_classification_non_ironic(masterpunct_df, summary_indiv_punct, 7, 6, 1)
nonironic_hash_dist_length = get_classification_non_ironic(masterpunct_df, summary_indiv_punct,4, 3, 8)
nonironic_star_dist = get_classification_non_ironic(masterpunct_df, summary_indiv_punct, 10, 9, 1)
nonironic_sademoji_dist = get_classification_non_ironic(masterpunct_df, summary_indiv_punct, 25, 24, 1)
# nonironic_leftsquare_dist_length = get_classification_non_ironic(masterpunct_df, summary_indiv_punct, 30, 29, 8)


# In[ ]:


# Add to IRONIC PREDICTOR DATAFRAME
non_ironic_predictor_df["!"] = nonironic_exclam_dist
non_ironic_predictor_df["'"] = nonironic_apost_dist_length
non_ironic_predictor_df["#"] = nonironic_hash_dist_length
non_ironic_predictor_df["*"] = nonironic_star_dist
non_ironic_predictor_df[":("] = nonironic_sademoji_dist
# non_ironic_predictor_df["["] = nonironic_leftsquare_dist_length

non_ironic_predictor_df = non_ironic_predictor_df.replace(np.nan, 0)
non_ironic_predictor_df.head()


# # (4) Classification Results

# ### (4.1) Feature weight against ironic df

# In[ ]:


#calculate the sum of all features for each comment
ironic_feature_prediction = ironic_predictor_df.sum(axis=1)

#add final column to ironic predictor df with feature totals
ironic_predictor_df["Feature Weight"] = ironic_feature_prediction

ironic_predictor_df = ironic_predictor_df.replace(np.nan, 0)
ironic_predictor_df.head()


# ### (4.2) Feature weight against non-ironic df

# In[ ]:


#calculate the sum of all features for each comment
non_ironic_feature_prediction = non_ironic_predictor_df.sum(axis=1)

#add final column to ironic predictor df with feature totals
non_ironic_predictor_df["Feature Weight"] = non_ironic_feature_prediction

non_ironic_predictor_df = non_ironic_predictor_df.replace(np.nan, 0)
non_ironic_predictor_df.head()

# non_ironic_feature_prediction


# ### (4.3) Final predictor label generation 
# #assign predictor labels to each comment
# #save to csv

# In[ ]:


#create final df with final predicitons
final_predictordf = pd.DataFrame(non_ironic_feature_prediction)

final_predictordf.columns = ["Non-ironic Feature Result"] + final_predictordf.columns.tolist()[1:]
final_predictordf["Ironic Feature Result"] = ironic_feature_prediction

final_predictordf.head()


# In[ ]:


#save test feature weight scores as csv
final_predictordf.to_csv(file_directory / "testing_featureweight_results.csv")


# In[ ]:


final_prediction = final_predicition_results(final_predictordf)

test["Prediction"] = final_prediction
print(len(val))

# #change order of columns (so label and prediction side by side)
test = test[['Comment_Text','Label','Prediction']]

#test.dtypes
test.head()


# In[ ]:


#save final test classification scores as csv
test.to_csv(file_directory / "testing_classification_results.csv")


# # (5) Accuracy Score

# ### (5.1) Calculate number of correctly predicted labels 

# In[ ]:


accuracy = accuracy(test)
print(accuracy)

