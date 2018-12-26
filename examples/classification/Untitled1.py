#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fasttext


# In[6]:


import re
from nltk.corpus import stopwords
regex = re.compile("[А-Яа-я]+")

mystopwords = stopwords.words('russian') + ['это', 'наш' , 'тыс', 'млн', 'млрд', 'также',  'т', 'д']

from pymystem3 import Mystem

m = Mystem()

def words_only(text, regex=regex):
    try:
        return " ".join(regex.findall(text))
    except:
        return ""

def remove_stopwords(text, mystopwords = mystopwords):
    try:
        return " ".join([token for token in text.split() if not token in mystopwords])
    except:
        return ""

def lemmatize(text, mystem=m):
    try:
        return "".join(m.lemmatize(text)).strip()  
    except:
        return " "


# In[10]:


classifier = fasttext.supervised('data.train.txt', 'model')


# In[16]:


content = [line.rstrip('\n') for line in open('input.txt')]
tmp = " ".join(content)
tmp = lemmatize(tmp)
tmp = tmp.lower()
tmp = words_only(tmp)
tmp = remove_stopwords(tmp)


# In[17]:


pred = classifier.predict_proba([tmp])
print('classifier =', pred)


# In[ ]:




