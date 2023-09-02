from flask import Flask,request,render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)




app=Flask(__name__)
app.static_folder = 'templates'

sw = pickle.load(open('stop.pkl' , 'rb'))
model = pickle.load(open('classifier.pkl','rb'))
vecto = pickle.load(open('vect.pkl','rb'))
binarizer = pickle.load(open('multibin.pkl','rb'))



@app.route ('/')
def home():
    return render_template('input.html')

@app.route ('/inputForm',methods = ['GET','POST'])



def pred () :

    try:
        input = request.form.get('inputString')
        op = remove_stopwords(input)
        op_vec = vecto.transform([op])
        pred_prob = model.predict(op_vec)

        t = 0.3
        predp = (pred_prob >= t).astype(int)
        ans = binarizer.inverse_transform(predp)
        return render_template('input.html' , inpu = input ,Output = ans)
        
    except:
        return render_template('input.html' , Output = "Invalid Input")



if __name__ == '__main__':
    
    app.run()