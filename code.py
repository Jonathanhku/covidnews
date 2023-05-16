#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[ ]:


c1 = pd.read_csv('news_dataset.csv', encoding= 'unicode_escape')


# In[ ]:


c1 = c1.dropna(axis=0)


# In[ ]:


import re
stop_words = set(stopwords.words('english'))
def clean(text):
    # Lowering letters
    text = text.lower()
    
    # Removing html tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Removing twitter usernames
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    
    # Removing urls
    text = re.sub('https?://[A-Za-z0-9]','',text)
    
    # Removing numbers
    text = re.sub('[^a-zA-Z]',' ',text)
    
    word_tokens = word_tokenize(text)
    
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)
    
    # Joining words
    text = (' '.join(filtered_sentence))
    return text


# In[ ]:


c1['clean_text'] = c1['post_text'].apply(clean)


# In[ ]:


c1.to_csv('clean_orginal.csv')


# RQ1

# In[ ]:


true = c1[c1['group'] == 0]
fake = c1[c1['group'] == 1]


# In[ ]:


a1 = true.sample(n=5000, random_state = 10)
a2 = fake.sample(n=5000, random_state = 10)


# In[ ]:


X = pd.concat([a1,a2], axis = 0)
X = X.sample(frac = 1, random_state = 10)


# In[ ]:


num = X[['like_count','retweet_count','n_char']]


# In[ ]:


def fivenumbers(df):
    d1 = pd.DataFrame({'index':['avg','min','max','sd']})
    for i in df.columns:
        if str(df[i].dtypes) == 'int32':
            avg = round(np.mean(df[i]),3)
            min1 = np.min(df[i])
            max1 = np.max(df[i])
            sd = round(np.std(df[i]),3)
            d1[i] = [avg, min1, max1, sd]
    d1 = d1.set_index('index')
    return d1


# In[ ]:


fivenumbers(X)


# In[ ]:


words = X['clean_text'].to_list()
groups = X['group'].to_list()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features = vectorizer.fit_transform(words)


# In[ ]:


from sklearn.feature_selection import SelectPercentile, f_classif
X_new = SelectPercentile(f_classif, percentile=50).fit_transform(features, groups)


# In[ ]:


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X_new, groups, test_size=0.2, random_state=10)


# In[ ]:


clf = MultinomialNB()
clf.fit(features_train,labels_train)


# In[ ]:


neg_class_prob_sorted = clf.feature_log_prob_[0, :].argsort()[::-1]
pos_class_prob_sorted = clf.feature_log_prob_[1, :].argsort()[::-1]

print(np.take(count_vect.get_feature_names(), neg_class_prob_sorted[:100]))
print(np.take(count_vect.get_feature_names(), pos_class_prob_sorted[:100]))


# In[ ]:


y_pred = clf.predict(features_test) 
precision_score(labels_test, y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(features_train,labels_train)
y_pred = clf.predict(features_test) 
precision_score(labels_test, y_pred)


# In[ ]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
#Creating the text variable
text = " ".join(cat for cat in fake.clean_text)


# In[ ]:


# Generate word cloud
word_cloud = WordCloud(
        width=3000,
        height=2000,
        random_state=1,
        background_color="salmon",
        colormap="Pastel1",
        collocations=False,
        stopwords=STOPWORDS,
        ).generate(text)


# In[ ]:


# Display the generated Word Cloud
plt.imshow(word_cloud)
plt.axis("off")
plt.savefig('wc.png')
plt.show()


# RQ2

# In[ ]:


c2 = c1.sample(n = 30000, random_state = 10)


# In[ ]:


c2 = c2.applymap(str)


# In[ ]:


from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = c2['clean_text'].to_list()
s_list = sentiment_pipeline(data)


# In[ ]:


list1 = []
for i in s_list:
    list1.append(i['label'])


# In[ ]:


c2['sentiment'] = list1


# In[ ]:


cl = c2.iloc[,2:]
cl.to_csv('sentiment.csv')


# In[ ]:


X = pd.read_csv('sentiment.csv')


# In[ ]:


X['like_count'] = X['like_count'].astype(int)
X['retweet_count'] = X['retweet_count'].astype(int)


# In[ ]:


predictor = X[['like_count','retweet_count']]
target = X['sentiment'].to_list()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(target)
target1 = le.transform(target)
target1


# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(predictor, target1, test_size=0.2, random_state=10)


# In[ ]:


import statsmodels.api as sm
log_reg = sm.Logit(labels_train,features_train).fit()


# In[ ]:


print(log_reg.summary())


# In[ ]:


log_reg.pvalues


# In[ ]:




