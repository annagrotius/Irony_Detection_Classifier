# Irony_RedditComments
### Final Assignment for Python

<b>The Dataset:</b> 
Reddit Comments pre-labelled with Ironic (-1) and Non-Ironic (1)

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


The XX has been structured in the following way:

<i>- Irony_classification_train</i>

<i>- Irony_classification_validation</i>

<i>- Irony_classification_test </i>

<i>- feature_stats.py </i>

<i>- classification_stats.py</i?


Train set summaries and visualisations


<i>- icsv </i>

<i>- icsv </i>


Validation set summaries and visualisations


<i>- icsv </i>

<i>- icsv </i>


* Test set summaries and visualisations


<i>- icsv </i>

<i>- icsv </i>


* Classification Results


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
            
            >>>> Note: This same file directory will be used for the entirety of the project. 
            
2) Download the files from this repository and remeber to save these in the same file directory as the dataset

3) Run the 'irony_train.py' file on your desired Python environment, remembering to insert your file directory path at the very         top.
            
            >>>> Note: All visualisations and tables should automatically be saved to your same file directory now.
   
4) Run the "irony_test.py" file on your desired Python environment, remembering again to insert the same file directory path at the very top.***

5) The results will be saved in your file directory under the name "test_classification_results.csv". The final accuracy score will be printed at the end of the script additionally.


*** A validation script is also included within this repository and once run, the classification results are automatically saved in your directory, as "validation_classification_results.csv". The validation stage was used to select the most relevant linguistic features extracted within the training set and thereafter adjust their weights. At the end of the script, the final accuracy score for the number of correctly predicted labels is printed. 

---
## FUNCTION INFORMATION:
Included within this repository are two files entitled "feature_stats.py" and "classification_stats.py", which include all the functions used within the project. Below will give you a brief overview of the two files***: 

### FEATURE_STATS:
This module include 13 functions, all of which are used within each of the subsets (train, validation & test). These are used to both preprocess the raw text, extract the specific linguistic features mentioned above, as well as return the releveant statistics.   

### CLASSIFICATION_STATS:
This module includes 4 functions, which are imported only within the validation and test subsets. Two are used to calculate the distance between the features to return the feature weight for each comment. The first one <i>get_classification_ironic</i>, performs the calculation against ironic comments (for row index [0]) and the second one <i>get_classification_non_ironic</i>, against non-ironic comments (for row index [1]).

![11](https://user-images.githubusercontent.com/46754140/52124124-84331b80-2628-11e9-91da-2441048c0c6b.JPG)

The third function, <i>final_predicition_results</i>, returns a final classification ("1" or -"1" for ironic or non-ironic) for the test subset. The final function, <i>accuracy</i> compares the prediction labels from previous functions and gold labels assigned to annotated comments and returns the accuracy of prediction labels.

*** For more information regarding all of the functions, including parameter inputs, please see the relevant python script. You can also use the <i>help(*insert_function_name)</i> function within your chosen environment.

---
For more information or any questions regarding the project, please contact the following person(s):

Lauren Green    lauren.cgreen@yahoo.com
Lisa Vasileva   liza.vasileva@gmail.com
