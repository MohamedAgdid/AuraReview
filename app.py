from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the entire pipeline (vectorizer + model)
pipeline = joblib.load('model/deployment_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
       
        prediction = pipeline.predict([review])[0]
        probabilities = pipeline.predict_proba([review])[0]
        
       
        sentiment_map = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"
        }
    
        conf_percent = {
            "Negative": round(probabilities[0]*100, 1),
            "Neutral": round(probabilities[1]*100, 1),
            "Positive": round(probabilities[2]*100, 1)
        }
        

        if prediction == 1:
            emotion = "😊"
            color = "success"
        elif prediction == -1:
            emotion = "😠"
            color = "danger"
        else:
            emotion = "😐"
            color = "warning"
        
        return render_template('result.html', 
                               review=review,
                               emotion=emotion,
                               color=color,
                               sentiment=sentiment_map[prediction],
                               conf_percent=conf_percent)

if __name__ == '__main__':
    # app.run(host='192.168.11.113', port=5000, debug=True)
    app.run(host='127.0.0.1', port=5000, debug=True)