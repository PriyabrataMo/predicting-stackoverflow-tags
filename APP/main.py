import io
import pickle

from flask import Flask,request,render_template
import pickle
import nltk
from nltk.corpus import stopwords

import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

from pydantic import BaseModel
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


model = pickle.load(open('classifier.pkl','rb'))
vecto = pickle.load(open('vect.pkl','rb'))
binarizer = pickle.load(open('multibin.pkl','rb'))

app = FastAPI()

app.add_middleware(CORSMiddleware, 
               allow_origins=['*'], 
               allow_credentials=True, 
               allow_methods=['*'], 
               allow_headers=['*'])


@app.post('/predict')
async def pred (req: Request):
    data = await req.json()
    input = data['question']
    try:
        op = remove_stopwords(input)
        op_vec = vecto.transform([op])
        pred_prob = model.predict(op_vec)

        t = 0.3
        predp = (pred_prob >= t).astype(int)
        ans = binarizer.inverse_transform(predp)
        print(ans)
        return {"input": input, "Output": ans[0]}
        
    except:
        return {"Output": "Invalid Input"}
