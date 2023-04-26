from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import joblib
scaler = joblib.load("A:/website/scalers.save")


app = Flask(__name__)
model = joblib.load('A:/website/modelss.pkl')

@app.route("/home")
@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

        feature_names = ["ph", "Hardness" , "Solids", "Chloramines", "Sulfate",
                         "Conductivity", "Organic_carbon","Trihalomethanes", "Turbidity"]

        df = pd.DataFrame(features_value, columns = feature_names)
        df = scaler.transform(df)
        output = model.predict(df)

        if output[0] == 1:           
            prediction = "aman"
        else:
            prediction = "tidak aman"


        return render_template('index.html', prediction_text= "Air ini {} untuk manusia konsumsi ".format(prediction))

        

if __name__ == "__main__":
    app.run(debug=True)