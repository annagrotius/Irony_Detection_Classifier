# Irony_RedditComments
Final Assignment for Python

The Dataset: Reddit Comments pre-labelled with Ironic (-1) and Non-Ironic (1)

Download: https://www.kaggle.com/rtatman/ironic-corpus "irony-labeled.csv" (193KB)

This classifier is trained to detect irony through extracting specifically chosen linguistic features, which can be seen below:
- (1) Average Word Count
- (2) Average Sentence Count
- (3) Punctuation Richness
- (4) Sarcasm Symbol
- (5) Upper-case Words
- (6) (Verb) Lemmas
- (7) Sentiment Classification (Textblob, Blobber; Naive Bayes)
- (8) Individual Punctuation Count
- (9) Word Type Count
- (10) Named Entity Count


In order to run this code, you will first need to download the packages/modules below; these can be loaded into your command line prompt. Links to where these can be downloaded from have been included:

SpaCy: (https://spacy.io/usage/)


TextBlob: (https://textblob.readthedocs.io/en/dev/install.html)

Once these have been downloaded, please proceed with the following steps:

---
STEPS:
1) Download the dataset (link above) and save in the file directory you wish.
            
            >>>> Note: This same file directory will be used for the entirety of the project. 
            
2) Download the files from this repository and remeber to save these in the same file directory as the dataset

3) Run the 'irony_train.py' file on your desired Python environment, remembering to insert your file directory path at the very         top.
            
            >>>> Note: All visualisations and tables should automatically be saved to your same file directory now.
   
4) Run the "irony_test.py" file on your desired Python environment, remembering again to insert the same file directory path at the very top.***

5) The results will be saved in your file directory under the name "test_classification_results.csv". The final accuracy score will be printed at the end of the script additionally.


*** A validation script is also included. This dataset was used to pick and choose features to include within the classifier and adjust their weighting. You can also use this script and associated dataset: once you have selected which linguistic features to include, run the rest of the code to attain the accuracy score, which will be shown at the very bottom. The results will be saved in your file directory under the name "validation_classification_results.csv".

---
FUNCTION INFORMATION
Included within this repository are two files entitled "feature_stats.py" and "classification_stats.py", which include all the functions used within the project. Below will give you a brief explanation regarding what each of them do and what parameters should be included.

*FEATURE_STATS:

get_all_tokens(test_dict)

get_words(listx)

get_punct(listx)

average_word_length(doc)

average_sent_length(doc)
check_sarcsymbol(doc)
count_uppercase(doc)
get_lemmas(doc)
get_punct_average(punctuation_list, token_comment_list)
get_lemmas(doc)
get_sentiment(dicts)
get_indiv_punct(doc)
relative_count_wordtypes(doc)
get_entities(doc)

*CLASSIFICATION_STATS:
get_classification_ironic(masterdf, newdf, mastercolumnindex_number, newcolumnindexnumber, weight)
get_classification_non_ironic(masterdf, newdf, mastercolumnindex_number, newcolumnindexnumber, weight)
final_predicition_results(feature_resultdf)
accuracy(testdf)

---
For more information or any questions regarding the project, please contact the following person(s):

Lauren Green    lauren.cgreen@yahoo.com
Lisa Vasileva   liza.vasileva@gmail.com
