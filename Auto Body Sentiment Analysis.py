
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df=pd.read_excel("C:/Users/psawant/Data/Auto Body Historical Data.xlsx")


df=df[df['Please rate the overall quality of the repair.'] !=3]


df['Positively Rated']=np.where(df['Please rate the overall quality of the repair.']>3,1,0)



df['Positively Rated'].mean()



df1 = df[['Comments / Feedback','Positively Rated']]



df1.dropna(inplace=True)



from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(df1['Comments / Feedback'],df1['Positively Rated'], random_state=0)


print('X_train first entry:\n',X_train[0])
print('\n X_train shape:',X_train.shape)


# ## Count Vectorizer


from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer().fit(X_train)


vect.get_feature_names()[::2000]


len(vect.get_feature_names())



X_train_vectorized=vect.transform(X_train)
X_train_vectorized



from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train_vectorized,y_train)


from sklearn.metrics import roc_auc_score
predictions=model.predict(vect.transform(X_test))
print('AUC',roc_auc_score(y_test,predictions))



feature_names=np.array(vect.get_feature_names())
sorted_coef_index=model.coef_[0].argsort()
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# ## Tf idf


from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())



X_train_vactorized=vect.transform(X_train)

model=LogisticRegression()
model.fit(X_train_vectorized,y_train)

predictions=model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test,predictions))



print(model.predict(vect.transform(['not an issue, car is working','an issue, caR is not working'])))


# ## n-grams

# In[87]:


vect=CountVectorizer(min_df=5,ngram_range=(1,2)).fit(X_train)
X_train_vectorized=vect.transform(X_train)

len(vect.get_feature_names())



model=LogisticRegression()
model.fit(X_train_vectorized,y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ',roc_auc_score(y_test,predictions))



feature_names=np.array(vect.get_feature_names())
sorted_coef_index=model.coef_[0].argsort()
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


print(model.predict(vect.transform(['not an issue, car is working','an issue, caR is not working'])))



df1['Comments / Feedback'] = df1['Comments / Feedback'].str.strip()




df1['Comments / Feedback']=df1['Comments / Feedback'].map(lambda x: x.strip())





