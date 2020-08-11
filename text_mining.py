#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Viktoria Akpan, Nicola Brioni   


#preprocessing the Reuters 21758 dataset

import string
from bs4 import BeautifulSoup
import re
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from glob import glob
import pandas as pd 
from pathlib import Path
import nltk
#uncomment this, if your computer doesnt have nltk. a popup window will download it
#nltk.download()
import bs4 as bs
#import urllib.request
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#!pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#read in multiple files 
directory_in_str = r'C:\Users\User\Downloads\ECONOMETRICS\AA DS PROJECT NLP\sgm_Files\sgm_Files\\'

def readInAllFilesFromFolder(directory_in_str):
    pathlist = Path(directory_in_str).glob('**\*.sgm')
    #initialize file where individual string texts will be added
    file = ""
    for path in pathlist:   
        path_in_str = str(path)
        fileX = open(path_in_str,'r')
        fileX = fileX.read()    
        file= file+ fileX
    return file
    
file = readInAllFilesFromFolder(directory_in_str)


document = BeautifulSoup(file,'html.parser')
all_articles = document.find_all('reuters')
topics_body = []

for article in all_articles:
      topics_body.append([str(article.topics), str(article.body)])

topics_body = [tup for tup in topics_body if tup[0] != "<topics></topics>" #deleting empty topics articles
               and tup[1] != "None"]   #deleting empty-body articles


def cleanTopic(topics_body):   #removing the html tags 
    
    for i in range(len(topics_body)):
        topics_body[i][0] = topics_body[i][0].replace("<topics>","")
        topics_body[i][0] = topics_body[i][0].replace("</topics>","")
        topics_body[i][0] = topics_body[i][0].replace("<d>","")
        topics_body[i][0] = topics_body[i][0].replace("</d>","")



cleanTopic(topics_body)

# cleaning bodies 
for i in range(len(topics_body)):  #making everything in lower case
     topics_body[i][1] = topics_body[i][1].lower()
    
for i in range(len(topics_body)):  #removing html tags + "reuter"
    topics_body[i][1] = topics_body[i][1].replace("<body>",'')
    topics_body[i][1] = topics_body[i][1].replace("</body>",'')



table = str.maketrans('', '', string.punctuation) #eliminating punctuation
for i in range(len(topics_body)):
    
    topics_body[i][1] = topics_body[i][1].translate(table)



#changing all numbers to "num"
for i in range(len(topics_body)):  
    topics_body[i][1] = re.sub(r'\d+', 'num', topics_body[i][1])




stopwords = set(stopwords.words('english') + ['reuter', '\x03', 'num', 'said']) #removing stopwords     
for i in range(len(topics_body)):
    #tokenize articles
    topics_body[i][1] = [word for word in topics_body[i][1].split() if word not in stopwords]

    
#rejoin words
stemmer = PorterStemmer() #stemming + turning articles back into strings
for i in range(len(topics_body)):
    topics_body[i][1] = " ".join([stemmer.stem(word) for word in topics_body[i][1]])
    

#make data list into a df, for easier manipulation
def put_dataList_into_df(topics_body):
    df = pd.DataFrame(topics_body) 
    df.columns = ['topic', 'text']
    df = df[['text', 'topic']]  
    return df

df = put_dataList_into_df (topics_body)
 
    

#FIND ALL MIXED TOPICS W 'MONEY-FX' IN IT 
listOfAllTopics = pd.unique(df.topic) 
# initializing substring 
substring_money = 'money'
substring_money_fx =  'money-fx'
# using filter() + lambda  to get string with substring  
multiple_topics_contains_money = list(filter(lambda x: substring_money in x, listOfAllTopics)) 
multiple_topics_contains_money_fx = list(filter(lambda x: substring_money_fx in x, listOfAllTopics)) 
not_money_fx_topics = list(set(listOfAllTopics) - set(multiple_topics_contains_money_fx))


#make binary cloumn of topic being about money-fx or containing it
def setBinaryColumnForListOfTopics(df, topicsToBe1): 
     #input is a list or a single topic in brackets as ['money-fx']
    df['binaryTopic']=0 
    for aTopic in topicsToBe1:
        df.loc[df.topic == aTopic, 'binaryTopic'] = 1
    return df

df =setBinaryColumnForListOfTopics(df, multiple_topics_contains_money_fx)#input is a list or a single topic in brackets as ['money-fx']


# In[2]:


#plot ratio of topics as binary classification  of the topic money-fx and other

sizeList=[]
#for aTopic in not_money_fx_topics:
sizeList.append ((df[df.binaryTopic == 0].text.count()))
    
sizeList.append ((df[df.binaryTopic == 1].text.count()))
#multiple_topics_contains_money_fx

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['other', 'money-fx']

#labels =  ListOfNOTMoney_fxTopicsW_minArticles #not_money_fx_topics
#sizes = [15, 30, 45, 10]
sizes = sizeList
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90) #, explode=explode
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[3]:


#word plot of all topics containing topic money-fx

#conculsion from word map: said and num were very common, so I added them to the stopwords to be taken out. after the call, 
# seperating it into types and looking at its effect came up, which seems like a good idea in later analyzis

my_df=df
neg_tweets = my_df[my_df.binaryTopic == 1]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[4]:


#all topics that are not money-fx
my_df=df
pos_tweets = my_df[my_df.binaryTopic == 0]
pos_string = []
for t in pos_tweets.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(pos_string)  #,colormap='magma'
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()


# In[5]:


#term frequency
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(my_df.text)


# In[6]:


import numpy as np

neg_doc_matrix = cvec.transform(df[df.binaryTopic==0].text)
pos_doc_matrix = cvec.transform(df[df.binaryTopic==1].text)

neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()


# In[7]:


#zips law needs total word count column to arange by
term_freq_df['total'] = term_freq_df[0] + term_freq_df[1]

print(term_freq_df.sort_values(by=0, ascending=False).iloc[:10])
print(term_freq_df.sort_values(by=1, ascending=False).iloc[:10])


# In[8]:


#zipf s law

y_pos = np.arange(100)
plt.figure(figsize=(10,8))
s = 1
expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
plt.bar(y_pos, term_freq_df.sort_values(by='total', ascending=False)['total'][:100], align='center', alpha=0.5,color='b')
plt.plot(y_pos, expected_zipf, color='r', linestyle='--',linewidth=2,alpha=0.5)
plt.ylabel('Frequency')
plt.title('Top 100 words in corpus')


# In[9]:


#make ML input w td-idf
from sklearn.feature_extraction.text import TfidfVectorizer


def createML_input_w_tfidf(df):

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')#, ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.text).toarray()
    y = df.binaryTopic
    return features,y
    
features,y = createML_input_w_tfidf(df)

def splitDataForTestTrain(features, y):
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, y, df.index, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test =splitDataForTestTrain(features, y)
    


# In[10]:


#model selection  
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

#needs  features, labels that are vectorized form of df
def plotModelAccuracies(features, y):
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, y, scoring='accuracy', cv=CV) ###make labels, feutures
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    import seaborn as sns
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    return cv_df, model_name
cv_df, model_name = plotModelAccuracies(features,y)


# In[11]:


#only works if cv df from prev function is a global var
cv_df.groupby('model_name').accuracy.mean() #probs its so low bc of so few data for some topics


# In[12]:


#confusion matrix
def makeConfusionMatrixOfSVC(features, y, df, model):
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, y, df.index, test_size=0.20, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred) 
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True)
                
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    return y_pred
y_pred_random_forest = makeConfusionMatrixOfSVC(features, y, df, RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0) )
y_pred_SVC = makeConfusionMatrixOfSVC(features, y, df,LinearSVC() )
y_pred_MultinomialNB = makeConfusionMatrixOfSVC(features, y, df, MultinomialNB() )
y_pred_logistic = makeConfusionMatrixOfSVC(features, y, df,LogisticRegression(random_state=0) )


# In[13]:


#check accuracy of each model
print("RandomForestClassifier"+ metrics.classification_report(y_test, y_pred_random_forest )) 
print("LinearSVC "+metrics.classification_report(y_test, y_pred_SVC )) 
print("MultinomialNB  "+metrics.classification_report(y_test, y_pred_MultinomialNB )) 
print("LogisticRegression"+metrics.classification_report(y_test, y_pred_logistic )) 

