# app.py

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import nltk

# Download NLTK data (only runs first time)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])

        return comment
    except:
        return comment


# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------
def load_model_and_vectorizer():
    mlflow.set_tracking_uri("http://3.88.87.182:5000")
    model_uri = "models:/yt_chrome_plugin_model/2"

    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load("./tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()


@app.route('/')
def home():
    return "Flask API Running ✅"


# -----------------------------
# SAFE PREDICTION FUNCTION
# -----------------------------
def predict_comments(comments):
    preprocessed = [preprocess_comment(c) for c in comments]

    # Vectorize
    transformed = vectorizer.transform(preprocessed)

    # Convert sparse matrix to DataFrame
    feature_names = vectorizer.get_feature_names_out()

    X_df = pd.DataFrame(
        transformed.toarray(),
        columns=feature_names
    )

    # Align with MLflow schema
    model_schema = model.metadata.get_input_schema()
    expected_columns = [col.name for col in model_schema.inputs]

    X_df = X_df.reindex(columns=expected_columns, fill_value=0)

    predictions = model.predict(X_df).tolist()
    predictions = [str(p) for p in predictions]

    return predictions


# -----------------------------
# PREDICT WITH TIMESTAMPS
# -----------------------------
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        predictions = predict_comments(comments)

        response = [
            {"comment": c, "sentiment": s, "timestamp": t}
            for c, s, t in zip(comments, predictions, timestamps)
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# -----------------------------
# PREDICT WITHOUT TIMESTAMP
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        predictions = predict_comments(comments)

        response = [
            {"comment": c, "sentiment": s}
            for c, s in zip(comments, predictions)
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# -----------------------------
# PIE CHART
# -----------------------------
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# -----------------------------
# WORDCLOUD
# -----------------------------
@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        preprocessed = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black'
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Word cloud failed: {str(e)}"}), 500


# -----------------------------
# TREND GRAPH
# -----------------------------
@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        monthly = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        plt.figure(figsize=(10, 5))

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value in monthly.columns:
                plt.plot(monthly.index, monthly[sentiment_value])

        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Trend graph failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)