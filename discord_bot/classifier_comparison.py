# Importing the libraries
import os.path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sn
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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Dense, MaxPooling1D, GlobalMaxPooling1D, Dropout, Flatten, LSTM

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

def classify(data, iteration):    
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

    tfid = TfidfVectorizer(vocabulary=mapping)
        
    X = tfid.fit_transform(trainer['Question']).toarray()
    y = trainer['YorN']

    classifier = SVC()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    print('fitting svc')
    classifier.fit(X_train, y_train)
    acc_svc = predictions(classifier, X_test, y_test)

    classifier = RandomForestClassifier(n_estimators = 50, max_depth = 50, n_jobs = -1)
    print('fitting RF')
    classifier.fit(X_train, y_train)
    acc_rf= predictions(classifier, X_test, y_test)

    classifier = LogisticRegression()
    print('fitting logistic')
    classifier.fit(X_train, y_train)
    acc_log= predictions(classifier, X_test, y_test)
    
    word_counter = buildVocab(data)
    X = [one_hot(ques, len(word_counter), lower = True) for ques in trainer['Question']]
    y = to_categorical(trainer['YorN'])

    max_len = 0
    for d in X:
        length = len(d)
        if max_len < length:
            max_len = length

    padded_ques = pad_sequences(X, maxlen = max_len, padding= 'post')
    X_train, X_test, y_train, y_test = train_test_split(padded_ques, y, test_size = 0.2)
    #print(padded_ques[0])

    model_conv = Sequential()
    model_conv.add(Embedding(input_dim = len(word_counter), output_dim = max_len, input_length = max_len))
    model_conv.add(Conv1D(filters = 100, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'))
    model_conv.add(Conv1D(filters = 100, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'))
    model_conv.add(MaxPooling1D(pool_size=2))
    model_conv.add(Flatten())
    model_conv.add(Dropout(0.2))
    model_conv.add(Dense(100, activation='relu'))
    model_conv.add(Dropout(0.5))
    model_conv.add(Dense(2,activation= 'softmax'))
    #print(model_conv.summary())

    model_conv.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    history_conv = model_conv.fit(X_train, y_train,
            batch_size=32,
            epochs=30,
            validation_split = 0.2,
            shuffle= True
            #callbacks = [callback]
            )

    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim = len(word_counter), output_dim = max_len, input_length = max_len))
    model_lstm.add(LSTM(100, return_sequences= True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(100))
    model_lstm.add(Dense(2, activation='softmax'))

    model_lstm.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    history_lstm = model_lstm.fit(X_train, y_train,
            batch_size=32,
            epochs=30,
            validation_split = 0.2,
            shuffle= True
            #callbacks = [callback]
            )

    pred_conv = model_conv.predict_classes(X_test)
    pred_lstm = model_lstm.predict_classes(X_test)

    from sklearn.metrics import confusion_matrix, accuracy_score
    y_test = np.argmax(y_test, axis = -1)
    cm_conv = confusion_matrix(y_test, pred_conv)
    cm_lstm = confusion_matrix(y_test, pred_lstm)
    acc_conv = accuracy_score(y_test, pred_conv)
    acc_lstm = accuracy_score(y_test, pred_lstm)
    print(cm_conv)
    print(acc_conv)
    print(cm_lstm)
    print(acc_lstm)
    store = pd.DataFrame(data={
        'iter': iteration,
        'SVC': acc_svc,
        'Random Forest': acc_rf,
        'Logistic Regression': acc_log,
        'Convolution NN': acc_conv,
        'LSTM NN': acc_lstm
    }, index=[iteration])
    keras.backend.clear_session()
    return store

data = pd.read_excel(r'files\questions.xlsx')
store = pd.DataFrame(columns=['iter','SVC','Random Forest','Logistic Regression', 'Convolution NN','LSTM NN'])
for i in range(0,11):
    store = store.append(classify(data,i))

store.reset_index(inplace=True)
store.to_csv('accuracy.csv')
#store = pd.read_csv('accuracy.csv')

plt.style.use('seaborn')
fig1 = plt.figure(figsize=(15,10))
ax = fig1.add_subplot(111)
ax.set_ylim(0.5,1)
plt.plot(store['SVC'],'r--o')
plt.plot(store['Random Forest'],'b--o')
plt.plot(store['Logistic Regression'],'g--o')
plt.plot(store['Convolution NN'],'y--o')
plt.plot(store['LSTM NN'],'c--o')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Test Accuracy of Various ML models')
plt.legend(['SVC','Random Forest','Logistic Regression','Convolution NN','LSTM NN' ])

x = ['SVC','Random Forest','Logistic Regression','Convolution NN','LSTM NN' ]
y = [round(store['SVC'].mean(),4),
    round(store['Random Forest'].mean(),4),
    round(store['Logistic Regression'].mean(),4),
    round(store['Convolution NN'].mean(),4),
    round(store['LSTM NN'].mean(),4)]

fig2 = plt.figure(figsize=(15,10))
ax = fig2.add_subplot(111)
ax.set_ylim(0.5,1)
plt.plot(x,y,'c-*')
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Mean Accuracy of ML Models')
for i,j in zip(x,y):
    ax.annotate(str(j),xy=(i,j+0.02))

fig1.savefig('iters.png')
fig2.savefig('model.png')