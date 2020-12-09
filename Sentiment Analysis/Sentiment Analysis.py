#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[448]:


import pandas as pd
import numpy as np
import glob, os, string, re, spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import six
from IPython.display import Image  
import pydotplus


# ## Import Datasets

# In[364]:


train_pos_files = glob.glob("aclImdb/train/pos/*.txt")
train_neg_files = glob.glob("aclImdb/train/neg/*.txt")
train_pos_ls = []

for i in train_pos_files:
    file = open(i, "r", encoding="utf8")
    str = file.readline()
    clean = re.compile('<.*?>')
    str = re.sub(clean, ' ', str)
    train_pos_ls.append(str)
    
train_neg_ls = []
for i in train_neg_files:
    file = open(i, "r", encoding="utf8")
    str = file.readline()
    clean = re.compile('<.*?>')
    str = re.sub(clean, ' ', str)
    train_neg_ls.append(str)


# In[365]:


labels = ['reveiw', 'label']
df_train_pos = pd.DataFrame()
df_train_pos['review'] = train_pos_ls
df_train_pos['label'] = 1
df_train_neg = pd.DataFrame()
df_train_neg['review'] = train_neg_ls
df_train_neg['label'] = -1
df_train = pd.concat([df_train_pos , df_train_neg])
df_train.head(10)


# In[366]:


test_pos_files = glob.glob("aclImdb/test/pos/*.txt")
test_neg_files = glob.glob("aclImdb/test/neg/*.txt")
test_pos_ls = []
for i in test_pos_files:
    file = open(i, "r",encoding="utf8")
    str = file.readline()
    clean = re.compile('<.*?>')
    str = re.sub(clean, ' ', str)
    test_pos_ls.append(str)
    
test_neg_ls = []
for i in test_neg_files:
    file = open(i, "r",encoding="utf8")
    str = file.readline()
    clean = re.compile('<.*?>')
    str = re.sub(clean, ' ', str)
    test_neg_ls.append(str)


# In[367]:


labels = ['reveiw', 'label']
df_test_pos = pd.DataFrame()
df_test_pos['review'] = test_pos_ls
df_test_pos['label'] = 1
df_test_neg = pd.DataFrame()
df_test_neg['review'] = test_neg_ls
df_test_neg['label'] = -1
df_test = pd.concat([df_test_pos , df_test_neg])
df_test.head(10)


# In[368]:


# Define text pre-processing functions
lemma = WordNetLemmatizer()
stops = set(stopwords.words('english'))
            
def text_prep(text):
    no_punct = [char for char in text if char not in string.punctuation]
    text = "".join(no_punct)
    text = [lemma.lemmatize(text, pos='v') for text in text.lower().split() if text not in stops] 
    text = " ".join(text)
    return (text)


# ## Data Preprocessing

# In[369]:


# preprocess training data
df_train['prep_review'] = df_train['review'].apply(lambda x:text_prep(x))
df_train[['prep_review', 'label']].head(10)


# In[370]:


# preprocess testing data
df_test['prep_review'] = df_test['review'].apply(lambda x:text_prep(x))
df_test[['prep_review', 'label']].head(10)


# In[461]:


# Vectorizing training data 
tfidf = TfidfVectorizer()
# tfidf = TfidfVectorizer(ngram_range = (1,3)) did not improve accuracy
x_train = tfidf.fit_transform(df_train['prep_review'])
y_train = df_train['label']


# In[462]:


# Vectorizing testing data 
x_test = tfidf.transform(df_test['prep_review'])
y_test = df_test['label']


# ## Training Model

# ### 1. Multinomial Naive Bayes

# In[463]:


mnb = MultinomialNB()


# In[464]:


mnb.fit(x_train, y_train)


# In[375]:


y_pred = mnb.predict(x_test)


# In[376]:


print(y_pred)


# In[377]:


accuracy = float(accuracy_score(y_test, y_pred))


# In[378]:


print("Accuracy Percentage of Multinomial Naive Bayes Model : %0.2f" % (accuracy*100) + '%')


# In[379]:


cm = confusion_matrix(y_train, y_pred)


# In[380]:


print("Confusion Matrix of Multinomial Naive Bayes Model -:")
print("True Negatives are : %d" % cm[0][0])
print("False Positives are : %d" % cm[0][1])
print("False Negatives are : %d" % cm[1][0])
print("True Positives are : %d" % cm[1][1])


# In[381]:


precision = float(precision_score(y_train, y_pred))


# In[382]:


print("Precision Percentage of Multinomial Naive Bayes Model : %0.2f" % (precision*100) + '%')


# In[383]:


rs = recall_score(y_train, y_pred)


# In[384]:


print("Recall Score Percentage of Multinomial Naive Bayes Model : %0.2f" % (rs*100) + '%')


# In[385]:


fs = f1_score(y_train, y_pred)


# In[386]:


print("F1 Score of Multinomial Naive Bayes Model : %0.2f" % (fs))


# ### 2. Random Forest Classifier

# In[387]:


rfc = RandomForestClassifier(n_estimators=100, random_state = 42, n_jobs = -1)


# In[388]:


rfc.fit(x_train, y_train)


# In[389]:


y_pred = rfc.predict(x_test)


# In[390]:


rfc.score(x_train, y_train)


# In[391]:


print(y_pred)


# In[392]:


accuracy = float(accuracy_score(y_test, y_pred))


# In[393]:


print("Accuracy Percentage of RandomForest Classifier Model : %0.2f" % (accuracy*100) + '%')


# In[394]:


cm = confusion_matrix(y_train, y_pred)


# In[395]:


print("Confusion Matrix of RandomForest Classifier Model -:")
print("True Negatives are : %d" % cm[0][0])
print("False Positives are : %d" % cm[0][1])
print("False Negatives are : %d" % cm[1][0])
print("True Positives are : %d" % cm[1][1])


# In[396]:


precision = float(precision_score(y_train, y_pred))


# In[397]:


print("Precision Percentage of RandomForest Classifier Model : %0.2f" % (precision*100) + '%')


# In[398]:


rs = recall_score(y_train, y_pred)


# In[399]:


print("Recall Score Percentage of RandomForest Classifier Model : %0.2f" % (rs*100) + '%')


# In[400]:


fs = f1_score(y_train, y_pred)


# In[401]:


print("F1 Score of RandomForest Classifier Model : %0.2f" % (fs))


# ### 3. Logistic Regression

# In[402]:


lr = LogisticRegression(solver = 'lbfgs', n_jobs = -1)


# In[403]:


lr.fit(x_train, y_train)


# In[404]:


y_pred = lr.predict(x_test)


# In[473]:


print(y_pred)


# In[475]:


lr.score(x_train, y_train)


# In[476]:


accuracy = float(accuracy_score(y_test, y_pred))


# In[408]:


print("Accuracy Percentage of LogisticRegression Model : %0.2f" % (accuracy*100) + '%')


# In[409]:


cm = confusion_matrix(y_train, y_pred)


# In[410]:


print("Confusion Matrix of LogisticRegression Model -:")
print("True Negatives are : %d" % cm[0][0])
print("False Positives are : %d" % cm[0][1])
print("False Negatives are : %d" % cm[1][0])
print("True Positives are : %d" % cm[1][1])


# In[411]:


precision = float(precision_score(y_train, y_pred))


# In[412]:


print("Precision Percentage of LogisticRegression Model : %0.2f" % (precision*100) + '%')


# In[413]:


rs = recall_score(y_train, y_pred)


# In[414]:


print("Recall Score Percentage of LogisticRegression Model : %0.2f" % (rs*100) + '%')


# In[415]:


fs = f1_score(y_train, y_pred)


# In[416]:


print("F1 Score of LogisticRegression Model : %0.2f" % (fs))


# ### 4. Linear Support Vector Classifier

# In[417]:


lsvm = LinearSVC()


# In[418]:


lsvm.fit(x_train, y_train)


# In[419]:


y_pred = lsvm.predict(x_test)


# In[420]:


print(y_pred)


# In[421]:


lsvm.score(x_train, y_train)


# In[422]:


accuracy = float(accuracy_score(y_test, y_pred))


# In[423]:


print("Accuracy Percentage of Linear Support Vector Classifier Model : %0.2f" % (accuracy*100) + '%')


# In[424]:


cm = confusion_matrix(y_train, y_pred)


# In[425]:


print("Confusion Matrix of Linear Support Vector Classifier Model -:")
print("True Negatives are : %d" % cm[0][0])
print("False Positives are : %d" % cm[0][1])
print("False Negatives are : %d" % cm[1][0])
print("True Positives are : %d" % cm[1][1])


# In[426]:


precision = float(precision_score(y_train, y_pred))


# In[427]:


print("Precision Percentage of Linear Support Vector Classifier Model : %0.2f" % (precision*100) + '%')


# In[428]:


rs = recall_score(y_train, y_pred)


# In[429]:


print("Recall Score Percentage of Linear Support Vector Classifier Model : %0.2f" % (rs*100) + '%')


# In[430]:


fs = f1_score(y_train, y_pred)


# In[431]:


print("F1 Score of Linear Support Vector Classifier Model : %0.2f" % (fs))


# ### 5. Decision Tree Classifier

# In[432]:


dtc = DecisionTreeClassifier(random_state=0)


# In[433]:


dtc.fit(x_train,y_train)


# In[434]:


y_pred = dtc.predict(x_test)


# In[435]:


print(y_pred)


# In[436]:


dtc.score(x_train, y_train)


# In[437]:


accuracy = float(accuracy_score(y_test, y_pred))


# In[438]:


print("Accuracy Percentage of Decision Tree Classifier Model : %0.2f" % (accuracy*100) + '%')


# In[439]:


cm = confusion_matrix(y_train, y_pred)


# In[440]:


print("Confusion Matrix of Decision Tree Classifier Model -:")
print("True Negatives are : %d" % cm[0][0])
print("False Positives are : %d" % cm[0][1])
print("False Negatives are : %d" % cm[1][0])
print("True Positives are : %d" % cm[1][1])


# In[441]:


precision = float(precision_score(y_train, y_pred))


# In[442]:


print("Precision Percentage of Decision Tree Classifier Model : %0.2f" % (precision*100) + '%')


# In[443]:


rs = recall_score(y_train, y_pred)


# In[444]:


print("Recall Score Percentage of Decision Tree Classifier Model : %0.2f" % (rs*100) + '%')


# In[445]:


fs = f1_score(y_train, y_pred)


# In[446]:


print("F1 Score of Decision Tree Classifier Model : %0.2f" % (fs))


# In[490]:


choice = int(input("Which Model You want to test by your Own Review?....\n1.Multinomial Naive Bayes\n2.Random Forest Classifier\n3.Logistic Regression\n4.Linear Support Vector Classifier\n5.Decision Tree Classifier\nEnter Choice..."))


# In[491]:


if choice == 1:
    predictions = mnb.predict(tfidf.transform([input("Enter Your Own Review :")]))[0]
elif choice == 2:
    predictions = rfc.predict(tfidf.transform([input("Enter Your Own Review :")]))[0]
elif choice == 3:
    predictions = lr.predict(tfidf.transform([input("Enter Your Own Review :")]))[0]
elif choice == 4:
    predictions = lsvm.predict(tfidf.transform([input("Enter Your Own Review :")]))[0]
elif choice == 5:
    predictions = dtc.predict(tfidf.transform([input("Enter Your Own Review :")]))[0]


if(predictions == -1):
	print("Negative Review!(-)")
else:
	print("Positive Review!(+)")


# In[ ]:




