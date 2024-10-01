from flask import Flask, render_template, request
import joblib
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


model = joblib.load('decision_tree_model.pkl')
le = LabelEncoder()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_name = request.form['user_name']
        user_location = request.form['user_location']
        user_description = request.form['user_description']
        text = request.form['text']
        hashtags = request.form['hashtags']
        retweets = int(request.form['retweets'])
        favorites = int(request.form['favorites'])

        
        user_name_enc = le.fit_transform([user_name])[0]  
        user_location_enc = le.fit_transform([user_location])[0]  
        user_description_enc = le.fit_transform([user_description])[0]  
        text_enc = le.fit_transform([text])[0]  
        hashtags_enc = le.fit_transform([hashtags])[0]  

        
        input_data = pd.DataFrame([[user_name_enc, user_location_enc, user_description_enc, text_enc, hashtags_enc, retweets, favorites]],
                                  columns=['user_name_enc', 'user_location_enc', 'user_description_enc', 'text_enc', 'hashtags_enc', 'retweets', 'favorites'])

     
        retweet_prediction = model.predict(input_data)[0] 

        
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  

        
        if sentiment > 0:
            sentiment_result = 'Positive'
        elif sentiment < 0:
            sentiment_result = 'Negative'
        else:
            sentiment_result = 'Neutral'

        
        return render_template('index.html', sentiment_result=sentiment_result, retweet_prediction=retweet_prediction)

if __name__ == '__main__':
    app.run(debug=True)