#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[154]:


import collections
import nltk.classify.util, nltk.metrics
from nltk import precision, recall
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.corpus import CategorizedPlaintextCorpusReader
from sklearn import svm
from sklearn.svm import LinearSVC
from nltk import precision
import string
from tabulate import tabulate         
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# ## Load Data

# In[155]:


train_path='aclImdb/train'
train_data=CategorizedPlaintextCorpusReader(train_path,r'(pos|neg)/.*\.txt',cat_pattern=r'(pos|neg)/.*\.txt')
test_path='aclImdb/test'
test_data=CategorizedPlaintextCorpusReader(test_path,r'(pos|neg)/.*\.txt',cat_pattern=r'(pos|neg)/.*\.txt')


# In[156]:


negative_train_id = train_data.fileids('neg')
positive_train_id = train_data.fileids('pos')
negative_test_id = test_data.fileids('neg')
positive_test_id = test_data.fileids('pos')


# In[157]:


def word_feats(words):
    return dict([(word, True) for word in words])


# ## Method 1

# In[158]:


negative_train = [(word_feats(train_data.words(fileids=[f])), 'neg') for f in negative_train_id]
positive_train = [(word_feats(train_data.words(fileids=[f])), 'pos') for f in positive_train_id]
negative_test = [(word_feats(test_data.words(fileids=[f])), 'neg') for f in negative_test_id]
positive_test = [(word_feats(test_data.words(fileids=[f])), 'pos') for f in positive_test_id]


# ## Train Model

# In[159]:


train_data = positive_train + negative_train


# In[160]:


test_data = positive_test + negative_test


# ### Naive Bayes Classification

# In[161]:


Naive_classifier = NaiveBayesClassifier.train(train_data)
refsets = collections.defaultdict(set)
testsets_Naive = collections.defaultdict(set)


# In[162]:


for i, (text, label) in enumerate(test_data):
        refsets[label].add(i)           
        observed_Naive = Naive_classifier.classify(text)
        testsets_Naive[observed_Naive].add(i)


# ### Accuracy and Precision

# In[163]:


accuracy = nltk.classify.util.accuracy(Naive_classifier, test_data)  
print("Accuracy of Naive Classifier Model: %0.2f" % (accuracy*100) + "%")
positive_precision = precision(refsets['pos'], testsets_Naive['pos'])
print("Precision of Positive Review of Naive Classifier Model: %0.2f" % (positive_precision*100) + "%")
positive_recall = recall(refsets['pos'], testsets_Naive['pos'])
negative_precision = precision(refsets['neg'], testsets_Naive['neg'])
print("Precision of Negative Review of Naive Classifier Model: %0.2f" % (negative_precision*100) + "%")
negative_recall = recall(refsets['neg'], testsets_Naive['neg'])


# In[164]:


Naive_classifier.show_most_informative_features(10)


# ### SVM Model
# 

# In[165]:


classifier = nltk.classify.SklearnClassifier(LinearSVC(max_iter=100000))
SVM_classifier = classifier.train(train_data)
refsets = collections.defaultdict(set)
SVM_testset = collections.defaultdict(set)


# In[166]:


for i, (text, label) in enumerate(test_data):
        refsets[label].add(i)           
        SVM_observe = classifier.classify(text)
        SVM_testset[SVM_observe].add(i)


# ### Accuracy and Precision

# In[167]:


accuracy = nltk.classify.util.accuracy(classifier, test_data)  
print("Accuracy of SVM Model: %0.2f" % (accuracy*100) + "%")
positive_precision = precision(refsets['pos'], SVM_testset['pos'])
print("Precision of Positive Review of SVM Model: %0.2f" % (positive_precision*100) + "%")
positive_recall = recall(refsets['pos'], SVM_testset['pos'])
negative_precision = precision(refsets['neg'], SVM_testset['neg'])
print("Precision of Negative Review of SVM Model: %0.2f" % (negative_precision*100) + "%")
negative_recall = recall(refsets['neg'], SVM_testset['neg'])


# ### Decision Tree Model

# In[168]:


negative_train_cutoff = len(negative_train)*1/100
positive_train_cutoff = len(positive_train)*1/100
train_Decision = negative_train[:int(negative_train_cutoff)] + positive_train[:int(positive_train_cutoff)]
DecisionTree_classifier = DecisionTreeClassifier.train(train_Decision)
refsets = collections.defaultdict(set)
Decision_Test = collections.defaultdict(set)


# In[169]:


for i, (text, label) in enumerate(test_data):
        refsets[label].add(i)           
        observed_Decision = DecisionTree_classifier.classify(text)
        Decision_Test[observed_Decision].add(i)


# ### Accuracy and Precision

# In[170]:


accuracy = nltk.classify.util.accuracy(DecisionTree_classifier, test_data)  
print("Accuracy of Decision Tree Model: %0.2f" % (accuracy*100) + "%")
positive_precision = precision(refsets['pos'], Decision_Test['pos'])
print("Precision of Positive Review of Decision Tree Model: %0.2f" % (positive_precision*100) + "%")
positive_recall = recall(refsets['pos'], Decision_Test['pos'])
negative_precision = precision(refsets['neg'], Decision_Test['neg'])
print("Precision of Negative Review of Decision Tree Model: %0.2f" % (negative_precision*100) + "%")
negative_recall = recall(refsets['neg'], Decision_Test['neg'])


# ## Method 2

# In[171]:


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    words_nopunc = [word for word in words if word not in string.punctuation]
    bigram_finder = BigramCollocationFinder.from_words(words_nopunc)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words_nopunc, bigrams)])


# In[172]:


train_path='aclImdb/train'
train_data=CategorizedPlaintextCorpusReader(train_path,r'(pos|neg)/.*\.txt',cat_pattern=r'(pos|neg)/.*\.txt')
test_path='aclImdb/test'
test_data=CategorizedPlaintextCorpusReader(test_path,r'(pos|neg)/.*\.txt',cat_pattern=r'(pos|neg)/.*\.txt')


# In[173]:


negative_train_id = train_data.fileids('neg')
positive_train_id = train_data.fileids('pos')
negative_test_id = test_data.fileids('neg')
positive_test_id = test_data.fileids('pos')


# In[174]:


negative_train = [(bigram_word_feats(train_data.words(fileids=[f])), 'neg') for f in negative_train_id]
positive_train = [(bigram_word_feats(train_data.words(fileids=[f])), 'pos') for f in positive_train_id]
negative_test = [(bigram_word_feats(test_data.words(fileids=[f])), 'neg') for f in negative_test_id]
positive_test = [(bigram_word_feats(test_data.words(fileids=[f])), 'pos') for f in positive_test_id]


# ## Train Model

# In[175]:


train_data = positive_train + negative_train


# In[176]:


test_data = positive_test + negative_test


# ### Naive Bayes Classification

# In[177]:


Naive_classifier = NaiveBayesClassifier.train(train_data)
refsets = collections.defaultdict(set)
testsets_Naive = collections.defaultdict(set)


# In[178]:


for i, (text, label) in enumerate(test_data):
        refsets[label].add(i)           
        observed_Naive = Naive_classifier.classify(text)
        testsets_Naive[observed_Naive].add(i)


# ### Accuracy and Precision

# In[179]:


accuracy = nltk.classify.util.accuracy(Naive_classifier, test_data)  
print("Accuracy of Naive Classifier Model: %0.2f" % (accuracy*100) + "%")
positive_precision = precision(refsets['pos'], testsets_Naive['pos'])
print("Precision of Positive Review of Naive Classifier Model: %0.2f" % (positive_precision*100) + "%")
positive_recall = recall(refsets['pos'], testsets_Naive['pos'])
negative_precision = precision(refsets['neg'], testsets_Naive['neg'])
print("Precision of Negative Review of Naive Classifier Model: %0.2f" % (negative_precision*100) + "%")
negative_recall = recall(refsets['neg'], testsets_Naive['neg'])


# ### Decision Tree Model
# 

# In[180]:


negative_train_cutoff = len(negative_train)*1/100
positive_train_cutoff = len(positive_train)*1/100
train_Decision = negative_train[:int(negative_train_cutoff)] + positive_train[:int(positive_train_cutoff)]
DecisionTree_classifier = DecisionTreeClassifier.train(train_Decision)
refsets = collections.defaultdict(set)
Decision_Test = collections.defaultdict(set)


# In[181]:


for i, (text, label) in enumerate(test_data):
        refsets[label].add(i)           
        observed_Decision = DecisionTree_classifier.classify(text)
        Decision_Test[observed_Decision].add(i)


# ### Accuracy and Precision
# 

# In[182]:


accuracy = nltk.classify.util.accuracy(DecisionTree_classifier, test_data)  
print("Accuracy of Decision Tree Model: %0.2f" % (accuracy*100) + "%")
positive_precision = precision(refsets['pos'], Decision_Test['pos'])
print("Precision of Positive Review of Decision Tree Model: %0.2f" % (positive_precision*100) + "%")
positive_recall = recall(refsets['pos'], Decision_Test['pos'])
negative_precision = precision(refsets['neg'], Decision_Test['neg'])
print("Precision of Negative Review of Decision Tree Model: %0.2f" % (negative_precision*100) + "%")
negative_recall = recall(refsets['neg'], Decision_Test['neg'])


# In[ ]:





# In[ ]:





# In[ ]:




