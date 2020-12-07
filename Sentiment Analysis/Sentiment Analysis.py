#!/usr/bin/env python
# coding: utf-8

# # SENTIMENT ANALYSIS

# ## Importing Libraries

# In[47]:


import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[48]:


REPLACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


# ## Load Dataset

# ### Training Set

# In[49]:


training_set = []
for line in open('data/movie_data/full_train.txt', 'r', encoding = "utf8"):
	training_set.append(line.strip())
print(training_set[0:10])


# In[50]:


testing_set = []
for line in open('data/movie_data/full_test.txt', 'r', encoding = "utf8"):
	testing_set.append(line.strip())
print(testing_set[0:10])


# ## Data Cleaning

# ### Training Set Cleaning 

# In[51]:


training_set = [REPLACE.sub("", line.lower()) for line in training_set]
training_set = [REPLACE_SPACE.sub(" ", line) for line in training_set]


# ### Testing Set Cleaning

# In[52]:


testing_set = [REPLACE.sub("", line.lower()) for line in testing_set]
testing_set = [REPLACE_SPACE.sub(" ", line) for line in testing_set]


# ## Train Model

# ### Count Vectorizer Model

# In[53]:


cv = CountVectorizer(binary=True)
cv.fit(training_set)
X = cv.transform(training_set)
X_test = cv.transform(testing_set)
target = [1 if i < 12500 else 0 for i in range(25000)]


# ## Splitting Data

# In[54]:


x_train, x_test , y_train, y_test = train_test_split(X, target, train_size = 0.75)


# ### Logistic Regression Model

# ## Calculating Accuracy_Score

# In[55]:


score = 0
for c in [0.01, 0.05, 0.25, 0.50, 0.75, 1.00]:
	lr = LogisticRegression(C=c,max_iter=100000)
	lr.fit(x_train, y_train)
	print ("Accuracy for C=%s: %s" 
	       % (c, accuracy_score(y_test, lr.predict(x_test))))
	if(score < accuracy_score(y_test, lr.predict(x_test))):
	    score = accuracy_score(y_test, lr.predict(x_test))


# In[56]:


output_model = LogisticRegression(C=score, max_iter=10000)
output_model.fit(X, target)


# ## Output Model Accuracy 

# In[57]:


accuracy = accuracy_score(target, output_model.predict(X_test))
percentage_accuracy = accuracy*100.00
print("Output Model Accuracy : %0.2f" % percentage_accuracy + "%")


# ## Examples

# In[58]:


feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), output_model.coef_[0]
    )
}


# In[59]:


print("Example of Positive words and it's Weightage")
for positive in sorted(
	feature_to_coef.items(), 
	key=lambda x: x[1], 
	reverse=True)[:5]:
	print (positive)


# In[60]:


print("Example of Negative words and it's Weightage")
for negative in sorted(
	feature_to_coef.items(), 
	key=lambda x: x[1])[:5]:
	print (negative)


# ## Test Human Generated Preview 

# In[ ]:


predictions = output_model.predict(cv.transform([input("Enter Your Own Review :")]))[0]
if(predictions == 0):
	print("Negative Review!(-)")
else:
	print("Positive Review!(+)")


# In[ ]:





# In[ ]:





# In[ ]:




