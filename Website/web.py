"""
Created on Wed Mar 10 18:58:55 2021

@author: Abhijith
"""
import  numpy as np
import pickle

from flask import Flask,render_template,request,url_for
app = Flask(__name__)

model=pickle.load(open('data/model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/prediction_page',methods=['GET','POST'])
def prediction_page():
    return render_template('predict.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    age             =   request.form['age']
    workclass       =   request.form['workclass']
    education       =   request.form['education']
    education_num   =   request.form['education_num']
    marital_status  =   request.form['marital_status']
   
    occupation      =   request.form['occupation']
    relationship    =   request.form['relationship']
    race            =   request.form['race']
    sex             =   request.form['sex']
    
    capitalgain     =   request.form['capital_gain']
    capitalloss     =   request.form['capital_loss']
    hoursperweek    =   request.form['hours_per_week']
    nativecountry   =   request.form['native_country']
    
    values=[age,workclass,education,education_num,marital_status,occupation,relationship,race,
            sex,capitalgain,capitalloss,hoursperweek,nativecountry]
    
    data=[]
    for i in values:
        data.append(i)
    
    result=np.array(data).reshape(1,-1)
    prediction_result=model.predict(result)
    output=prediction_result.item()
    
    return render_template('result.html',predicted_value = format(output))

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run()

