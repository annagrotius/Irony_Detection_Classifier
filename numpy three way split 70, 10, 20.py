
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


# In[2]:


filename= "/Users/laure/OneDrive/Dokumente/VU/Python for Text Analysis/Final Assignment/irony-labeled.csv"

df = pd.read_csv(filename)
df.head()


# In[3]:


#checking for any missing values
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 

    
#ALTERNATIVELY    
#add ".sum" to get total number (if applic.)
df.isnull().sum()


# In[43]:


#split between train and test data
#with validation

train, val, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.8*len(df))])
print(f"Validation Size {len(val)}")
print(f"Train Size {len(train)}")
print(f"Test Size {len(tes)}")


# In[44]:


print("Comment_label:", train["comment_text"].iloc[:4])
print("Label:", train["label"].iloc[:4])


print('Training Data Shape:', train.shape)
print('Testing Data Shape:', test.shape)


# In[45]:


#summarises the distribution of comments by different labels 
fig = plt.figure(figsize=(8,4))
sns.barplot(x = train["label"].unique(), 
            y=train["label"].value_counts())

plt.title("Training Data: Ironic vs Non-ironic Reddit Comments")
plt.ylabel("Number of Reddit Comments")
plt.xlabel("Irony Labels (Ironic=1; Not Ironic=-1 )")
plt.show()


# In[47]:


#VALIDATION
fig = plt.figure(figsize=(8,4))
sns.barplot(x = val["label"].unique(), 
            y=val["label"].value_counts())

plt.title("Validation Data: Ironic vs Non-ironic Reddit Comments")
plt.ylabel("Number of Reddit Comments")
plt.xlabel("Irony Labels (Ironic=1; Not Ironic=-1 )")
plt.show()


# In[48]:


#TESTING
fig = plt.figure(figsize=(8,4))
sns.barplot(x = test["label"].unique(), 
            y=test["label"].value_counts())

plt.title("Testing Data: Ironic vs Non-ironic Reddit Comments")
plt.ylabel("Number of Reddit Comments")
plt.xlabel("Irony Labels (Ironic=1; Not Ironic=-1 )")
plt.show()


# In[7]:


import spacy

nlp = spacy.load("en_core_web_sm")
punct = string.punctuation

for x in df:
    doc = nlp(x)

