# Irony_RedditComments
### Final Assignment for Python

This project is a final assignment for the "Python for Text Analysis" course at Vrije Universiteit (VU), Amstedam, The Netherlands. The goal of this text classification project is to build a classifier for irony detection, with Reddit Comments as the dataset. 

---
<b>The Dataset:</b> Reddit Comments pre-labelled with Ironic (-1) and Non-Ironic (1) tags.

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

----
## NAVIGATION:
To navigate across the project files on Github, the project has been structured in the following way:

<i>- Irony_classification_train.py</i>

<i>- Irony_classification_validation.py</i>

<i>- Irony_classification_test.py </i>

<i>- feature_stats.py </i>

<i>- classification_stats.py</i>


### Train set summaries and visualisations [Folder]


<i>- train_summary_general.csv </i> [csv file]

<i>- train_summary_pos.csv </i> [csv file]

<i>- train_summary_namedentity.csv </i> [csv file]

<i>- train_summary_puncttype.csv </i> [csv file]

<i>- train_summary_general.png </i> [image]

<i>- train_summary_POS.png </i> [image]

<i>- train_summary_NER.png </i> [image]

<i>- train_summary_punctuation.png </i> [image]



### Validation set summaries and visualisations [Folder]



<i>- validation_summary_general.csv </i> [csv file]

<i>- validation_summary_pos.csv </i> [csv file]

<i>- validation_summary_namedentity.csv </i> [csv file]

<i>- validation_summary_puncttype.csv </i> [csv file]



### Test set summaries and visualisations [Folder]

<i>- test_summary_general.csv </i> [csv file]

<i>- test_summary_pos.csv </i> [csv file]

<i>- test_summary_namedentity.csv </i> [csv file]

<i>- test_summary_puncttype.csv </i> [csv file]



### Classification Results [Folder]


<i>- validation_featureweights_result.csv </i>

<i>- validation_classification_result.csv </i>

<i>- test_featureweights_result.csv </i>

<i>- test_classification_result.csv </i>



---

In order to run this code, you will first need to download the packages/modules below; these can be loaded into your command line prompt. Links to where these can be downloaded from have been included:

<i>SpaCy:</i> (https://spacy.io/usage/)


<i>TextBlob:</i> (https://textblob.readthedocs.io/en/dev/install.html)

Once these have been downloaded, please proceed with the following steps:

---
## STEPS: 
1) Download the dataset (link above) and save in the file directory you wish.
            
            >>>> Note: This same file directory should be used for the entirety of the project. 
            
2) Download the python scripts from this repository in the main section and remeber to save these in the same file directory as the dataset.

3) Run the 'irony_train.py' file on your desired Python environment, remembering to insert your file directory path at the very         top.
            
            >>>> Note: All visualisations and tables should automatically be saved to your same file directory now.
   
4) Run the "irony_test.py" file on your desired Python environment, remembering again to insert the same file directory path at the very top.***<i>see below</i>

5) The results will be saved in your file directory under the name "test_classification_results.csv". The final accuracy score will be printed at the end of the script additionally.


*** A validation script is also included within this repository and once run, the classification results are automatically saved in your directory, as "validation_classification_results.csv". The validation stage was used to select the most relevant linguistic features extracted within the training set and thereafter adjust their weights. At the end of the script, the final accuracy score for the number of correctly predicted labels is printed. 

---
## FUNCTION INFORMATION:
Included within this repository are two files entitled "feature_stats.py" and "classification_stats.py", which include all the functions used within the project. Below will give you a brief overview of the two files***: 

### FEATURE_STATS:
This module include 13 functions, all of which are used within each of the subsets (train, validation & test). These are used to both preprocess the raw text, extract the specific linguistic features mentioned above, as well as return the releveant statistics.   

### CLASSIFICATION_STATS:
This module includes 4 functions, which are imported only within the validation and test subsets. Two are used to calculate the distance between the features to return the feature weight for each comment. The first one <i>get_classification_ironic</i>, performs the calculation against ironic comments (for row index [0]) and the second one <i>get_classification_non_ironic</i>, against non-ironic comments (for row index [1]). Please see below for the applicable formula:

![classification_formula](https://user-images.githubusercontent.com/44449955/52131227-8e5f1500-263c-11e9-9949-2b429412b805.PNG)

The third function, <i>final_predicition_results</i>, returns a final classification ("1" or -"1" for ironic or non-ironic) for the test subset, which has been based on the formula below:

![finalprediction_formula](https://user-images.githubusercontent.com/44449955/52131095-3b855d80-263c-11e9-8916-e038740e8292.PNG)

The final function, <i>accuracy</i> compares the prediction labels from previous functions and gold labels assigned to annotated comments and returns the accuracy of prediction labels.

*** For more information regarding all of the functions, including parameter inputs, please see the relevant python script. You can also use the <i>help(*insert_function_name)</i> function within your chosen environment.

---
For more information or any questions regarding the project, please contact the following person(s):

Lauren Green    lauren.cgreen@yahoo.com
Lisa Vasileva   liza.vasileva@gmail.com
