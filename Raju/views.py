from django.db.models.fields import EmailField
from Raju import urls
from django.shortcuts import redirect, render
from django.contrib.auth import authenticate, login, logout
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse
from . import models,serializers
from .serializers import new
from django.shortcuts import render
from newsapi import NewsApiClient
import json
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
import pandas as pd
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import os



@api_view(['POST'])
def texttoemo(request):
    reqdata =  request.data
    df = pd.read_csv("C:/Users/admin/Downloads/emotion_dataset_raw.csv")
    df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
    Xfeatures = df['Clean_Text']
    ylabels = df['Emotion']
    x_train,y_train=Xfeatures,ylabels
    pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
    pipe_lr.fit(x_train,y_train)
    ex1 = str(reqdata["data"])
    data=pipe_lr.predict([ex1])
    tep, created = models.Informn.objects.get_or_create(description = reqdata["data"])
    tep.emotion = data[0]
    tep.save()
    return Response(data[0])



@api_view(['POST'])
def faketrue(request):
    df = pd.read_csv('C:/Users/admin/Downloads/fake_or_real_news/fake_or_real_news.csv')
    df['label'] = df['label'].apply(lambda x: 1 if x=='REAL' else 0)
    X_train, X_test, Y_train, Y_test = train_test_split(df.text, df.label, test_size = 0.25)

    v = TfidfVectorizer(stop_words='english', max_df =0.9)

    X_train_count = v.fit_transform(X_train.values)

    LogReg_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    DTree_clf = DecisionTreeClassifier()
    SVC_clf = SVC()

    LogReg_clf.fit(X_train_count, Y_train)
    DTree_clf.fit(X_train_count, Y_train)
    SVC_clf.fit(X_train_count, Y_train)
    
    voting_clf = VotingClassifier(estimators=[('SVC', SVC_clf), ('DTree', DTree_clf), ('LogReg', LogReg_clf)], voting='hard')
    voting_clf.fit(X_train_count, Y_train)

    reqdata =  request.data
    ex1 = str(reqdata["data"])

    ex1=[ex1]

    ex1_count = v.transform(ex1)

    preds = voting_clf.predict(ex1_count)

    if(preds==0):
        return Response(["Fake"])
    else :
        return Response(["Not Fake"])


@api_view(['POST'])
def posneg(request):
    
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder 

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    padded_shapes = ([None], ())

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes = padded_shapes)
    test_dataset = test_dataset.padded_batch(BATCH_SIZE,padded_shapes = padded_shapes)

    def pad_to_size(vec, size):
        zeroes = [0]*(size - len(vec))
        vec.extend(zeroes)
        return vec

    def sample_predict(loaded_model,sentence, pad):
        encoded_sample_pred_text = encoder.encode(sentence)
        if pad:
            encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
        encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
        predictions = loaded_model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

        return predictions

    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])

    model.fit(train_dataset, epochs=10,validation_data=test_dataset,validation_steps=30)

    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('model.h5')

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics = ['accuracy'])

    reqdata =  request.data
    ex1 = str(reqdata["data"])
    ex1=(ex1)
    
    predictions = sample_predict(loaded_model,ex1, pad=True)*100
    
    if(float(predictions)>=0.00):
        return Response(["Positive"])
    else:
        return Response(["Negative"])


@api_view(['POST'])
def register_fn(request):
    reqdata = request.data
    user, created = User.objects.get_or_create(username=reqdata['email'])
    if created :
        try :
            user.set_password(reqdata['password'])
            user.save()
            usera,create = models.UserItem.objects.get_or_create(email=reqdata['email'])
            usera.name = reqdata['name']
            usera.save()
            token = Token.objects.create(user=user)
            first_serializer = new(usera, many=False)
            return Response({
                'reqdata':reqdata,
                'ser': first_serializer.data,
                'token': token.key,
            })

        except :
            user.delete()
            return Response({'error in process'})
    else :
        return Response({
            'user':user.id,
            'status': 'already created',
        })

@api_view(['POST'])
def login_fn(request):
    reqdata=request.data
    useremail=reqdata['email']
    userpass=reqdata['password']
    try:
        user = User.objects.get(username=useremail)
        user.check_password(userpass)
        token, created = Token.objects.get_or_create(user=user)
        return Response({'token':token.key})
    except:
        return Response({'status':'No user exists'})


@api_view(['GET'])
def me(request):
    news=NewsApiClient(api_key='d7bbd6611e2d4e9fba97d1410a5c0b7d')
    data=news.get_sources()
    data=data["sources"]
    return Response(data)

@api_view(['GET'])
def he(request):
    news=NewsApiClient(api_key='d7bbd6611e2d4e9fba97d1410a5c0b7d')
    data=news.get_sources()
    data=data['sources']
    p=[]
    for i in data:
        p.append(i['category'])
    p=set(p)
    p=list(p)
    p=json.dumps(p)
    return HttpResponse(p)


@api_view(['POST'])
def get_new(request):
    cat = request.POST["cat"]
    news=NewsApiClient(api_key='d7bbd6611e2d4e9fba97d1410a5c0b7d')
    data=news.get_top_headlines(category=cat,language='en')
    data=data['articles']
    return Response(data)