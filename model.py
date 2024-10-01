import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import joblib


data = pd.read_csv('vaccination_all_tweets.csv')


le = LabelEncoder()


data['user_name_enc'] = le.fit_transform(data['user_name'])
data['user_location_enc'] = le.fit_transform(data['user_location'])
data['user_description_enc'] = le.fit_transform(data['user_description'])
data['text_enc'] = le.fit_transform(data['text'])
data['hashtags_enc'] = le.fit_transform(data['hashtags'])


X = data.drop('is_retweet', axis=1)
y = data['is_retweet']  


features = X[['user_name_enc', 'user_location_enc', 'user_description_enc', 'text_enc', 'hashtags_enc', 'retweets', 'favorites']]


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


joblib.dump(model, 'decision_tree_model.pkl')
print("Model saved as decision_tree_model.pkl")


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  

    
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


data['sentiment_result'] = data['text'].apply(analyze_sentiment)


data['sentiment_encoded'] = data['sentiment_result'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})


print(data[['text', 'sentiment_result', 'sentiment_encoded']].head())


data.to_csv('vaccination_with_sentiment.csv', index=False)
print("Sentiment analysis complete. Results saved to 'vaccination_with_sentiment.csv'.")