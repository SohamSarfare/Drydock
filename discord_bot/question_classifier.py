# Importing the libraries
import os.path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import re
import copy
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
new_stopwords = set(stopwords.words('english')) - {'what', 'where', 'who', 
                                                   'when', 'which', 'does', 
                                                   'how', 'is','whatever',
                                                   'whomever', 'whose', 'why', 
                                                   'will'}
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# Cleaning the texts
def clean(df):
    for i in range(0,df.shape[0]):
        try:
            ques = re.sub(r'^https?:\/\/.*[\r\n]*', ' hyperlink ', df.loc[i,'Question'], flags=re.MULTILINE)
            ques = re.sub(r'[^@\?a-zA-Z]', ' ', ques, flags=re.MULTILINE)
            ques = re.sub(r'[\?]', ' question ', ques, flags=re.MULTILINE)
            ques = re.sub(r'[@]', ' answer ', ques, flags=re.MULTILINE)
            ques = re.sub(r'[^\w]', ' ', ques, flags=re.MULTILINE)
            ques = ques.lower()
            ques = ques.split()
            wn = WordNetLemmatizer()
            temp = pos_tag(ques)
            ques = [wn.lemmatize(word, tag_map[tag[0]]) for (word,tag) in temp if word not in new_stopwords]
            ques = ' '.join(ques)
            df.loc[i,'Question'] = ques.lower().strip()
        except:
            print(i)
    return df

def buildVocab(df):
    word_counter = Counter()
    for s in df['Question']:
        word_counter.update(s.split())
    return word_counter

def predictions(classifier, X_test, y_test):
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cm)
    print(acc)
    return acc

#def trainClassify():

def classify(data, video_id):    
    cleaned_data = data.copy(deep=True)
    data['Question'].replace('', np.nan, inplace=True)
    data.dropna(subset=['Question'], inplace=True)
    trainer = pd.read_excel(r'files\questions_trainer.xlsx')

    cleaned_data['Question'].replace('', np.nan, inplace=True)
    cleaned_data.dropna(subset=['Question'], inplace=True)
    cleaned_data = clean(cleaned_data)
    trainer = clean(trainer)
    mapping = {}
    if not os.path.isfile('map.pkl'):
        vocab = buildVocab(data)
        mapping = { w : i for i, w in enumerate(vocab) }
        joblib.dump(mapping, r'map.pkl')
    else:
        mapping = joblib.load('map.pkl')
    #print(vocab)
    acc = 0

    tfid = TfidfVectorizer(vocabulary=mapping)
    if os.path.isfile(r'svc.pkl'):
        classifier = joblib.load(r'svc.pkl')
        X = tfid.fit_transform(trainer['Question']).toarray()
        y = trainer['YorN']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
        acc = predictions(classifier, X_test, y_test)
        
        test = tfid.transform(cleaned_data['Question']).toarray()
        pred = classifier.predict(test)

        classified = pd.DataFrame(data={
            'User': data['User'],
            'Question': data['Question'],
            'Video': data['Video'],
            'VideoID': data['VideoID'],
            'YorN': pred
        })
        classified.to_excel(r'files\questions_classified_{}.xlsx'.format(video_id), index= False)
        print('Saved output')
    else:        
        X = tfid.fit_transform(trainer['Question']).toarray()
        y = trainer['YorN']

        classifier = SVC()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
        classifier.fit(X_train, y_train)
        acc = predictions(classifier, X_test, y_test)
        joblib.dump(classifier,r'svc.pkl')

        test = tfid.transform(cleaned_data['Question']).toarray()
        pred = classifier.predict(test)
        classified = pd.DataFrame(data={
            'User': data['User'],
            'Question': data['Question'],
            'Video': data['Video'],
            'VideoID': data['VideoID'],
            'YorN': pred
        })
        classified.to_excel(r'files\questions_classified_svc.xlsx', index= False)
        print('Saved output')
    return acc