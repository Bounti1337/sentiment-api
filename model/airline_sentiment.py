#%%
import pandas as pd
from lightgbm import LGBMClassifier
from mlxtend.classifier import LogisticRegression
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/alekseivarentsov/Downloads/archive/Tweets.csv')

print(
      data.shape , '\n' ,
      data.head() , '\n' ,
      data.nunique() , '\n' ,
      data.dtypes , '\n' ,
      data.describe() , '\n' ,
      )
#%%

#%%
import seaborn as sns
import matplotlib.pyplot as plt

isnull_data = data.isnull().sum()
isnull_data.sort_values( ascending = False).plot( kind = 'bar' )
plt.show()

obj_cols = []
num_cols = []

for i in data[isnull_data[isnull_data > 0].index]:
    if data[i].dtypes == object:
        obj_cols.append(i)
    else:
        num_cols.append(i)
#%%
# nice std so we can use mean to fill null
print(data[[i for i in num_cols]].describe().loc['std'])

for col in num_cols:
    data[col] = data[col].fillna( data[col].mean() )

for col in obj_cols:
    data[col] = (data[col].fillna('none'))
#%%
data.groupby('airline')['airline_sentiment'].count()
#%%
import re
from nltk.tokenize import TreebankWordTokenizer
import spacy
from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer


sample = data['text']
containers = {

    "i'm" : "i am",
    "i'd" : "i would" ,
    "i'll" : "i will",
    "i've" : "i have",

    "you're" : "you are",
    "you'll" : "you will",
    "you've" : "you have",

    "he's" : "he is",
    "he'll" : "he will",

    "she's" : "she is" ,
    "she'll" : "she will",

    "it's" : "it is",
    "it'll" : "it will",

    "isn't" : "is not",
    "aren't" : "are not",

    "didn't" : "did not" ,
    "don't" : "do not" ,
    "can't" : "can not" ,
    "couldn't" : "could not" ,

    "what's" : "what is",
    "what're" : "what are",

    "thx" : "thanks" ,
    "pls" : "please" ,

}

for value in set(data['airline']):
    containers[value.lower().replace(' ' , '' )] = ''

class prepocessing_data():
    def __init__(self):
        self.tfidf =  TfidfVectorizer( ngram_range=(1,2))
        self.lemmatizer = spacy.load('en_core_web_sm')

    def clean(self, text):
        text = text.lower()

        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@' , '' , text)
        text = re.sub(r'#' , '' , text)

        for prev_val , val in containers.items():
            text = re.sub(rf'\b{prev_val}\b' , val , text)

        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def lemma(self , word):
        doc = self.lemmatizer(word)
        lemmas = [token.lemma_ for token in doc ]
        return ' '.join(lemmas)

    def Tfidf(self , data):
        return self.tfidf.fit_transform(data)

    def transform_Tfidf(self , data):
        return self.tfidf.transform(data)

#%% md
# 
#%%
from sklearn.model_selection import train_test_split

X = data['text']
y = data['airline_sentiment']

X_train , X_test , y_train , y_test = train_test_split( X , y , test_size = 0.2 , random_state = 42)

preproc = prepocessing_data()
X_train = X_train.apply( preproc.clean )
X_test = X_test.apply( preproc.clean )

X_train = X_train.apply( preproc.lemma )
X_test = X_test.apply( preproc.lemma )
#%%
y_train = y_train.map({'negative' : 0 , 'neutral' : 1 , 'positive' : 2 })

y_test = y_test.map({'negative' : 0 , 'neutral' : 1 , 'positive' : 2 })

X_train = preproc.Tfidf(X_train)
X_test = preproc.transform_Tfidf(X_test)
#%%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    ).fit(X_train , y_train)
y_pred = model.predict_proba(X_test)


print(roc_auc_score(
    y_test,
    y_pred,
    multi_class='ovr',
    average='macro'
))
#%%
import joblib

tfidf = joblib.load('model/tfidf.pkl')
model = joblib.load('model/logreg_text.pkl')


