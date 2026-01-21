import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

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

# for value in set(data['airline']):
#     containers[value.lower().replace(' ' , '' )] = ''

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
