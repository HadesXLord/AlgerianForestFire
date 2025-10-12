from flask import Flask,jsonify,request,render_template
import joblib
import numpy as np
import pandas as pd


#load the model

linear_model=joblib.load('Algerian_forest_fire_project/models/linear_model.pkl')
logistic_model=joblib.load('Algerian_forest_fire_project/models/logistic_model.pkl')


application=Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict_linear',methods=['GET','POST'])

def predict_linear():
    if request.method=='POST':
        features=np.array([
            float(request.form.get("Temperature")),
            float(request.form.get("RH")),
            float(request.form.get("Ws")),
            float(request.form.get("Rain")),
            float(request.form.get("FFMC")),
            float(request.form.get("DMC")),
            float(request.form.get("DC")),
            float(request.form.get("ISI")),
            float(request.form.get("BUI"))]).reshape(1,-1)

        result=linear_model.predict(features)
        return render_template("predict_linear_home.html",result=result[0])
    else:
        return render_template("predict_linear_home.html")

@app.route("/predict_logistic",methods=['GET','POST'])

def predict_logistic():
    if request.method=="POST":
        features=np.array([
            float(request.form.get("Temperature")),
            float(request.form.get("RH")),
            float(request.form.get("Ws")),
            float(request.form.get("Rain")),
            float(request.form.get("FFMC")),
            float(request.form.get("DMC")),
            float(request.form.get("DC")),
            float(request.form.get("ISI")),
            float(request.form.get("BUI"))]).reshape(1,-1)

        pred=logistic_model.predict(features)[0]
        result="Fire chances HIGH 🔥" if pred==1 else "Fire chances LOW ✅"

       
        alert_class="alert-danger" if pred==1 else "alert-success"
        return render_template("predict_logistic_home.html",result=result,alert_class=alert_class)
    
    else:
        return render_template("predict_logistic_home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5050,debug=True)





