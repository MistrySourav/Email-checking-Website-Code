from flask import Flask, render_template, request  # type: ignore
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

def train_model():
    # Load dataset (using your original data loading approach)
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['Label', 'EmailText']].rename(columns={'Label': 'Label', 'EmailText': 'EmailText'})
    
    # Your original preprocessing steps
    df = df.copy()
    df['label_numeric'] = df['Label'].map({'spam': 1, 'ham': 0})
    df = df.drop_duplicates(keep='first')
    
    # Your original data splitting
    x = df['EmailText'].fillna('')
    y = df['label_numeric']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Your original vectorization approach
    tfidf1 = TfidfVectorizer()
    feature_train = tfidf1.fit_transform(x_train)
    
    # Your model training parameters
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(feature_train, y_train)
    
    return svm_model, tfidf1

# Load or create model (modified to use your training workflow)
try:
    svm_model = joblib.load('svm_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except:
    print("Training new model using your methodology...")
    svm_model, tfidf_vectorizer = train_model()
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

def classify_email(email_text):
    # Using your original prediction approach without additional preprocessing
    email_feature = tfidf_vectorizer.transform([email_text])
    predicted_label = svm_model.predict(email_feature)
    return 'spam' if predicted_label[0] == 1 else 'ham'

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        email_text = request.form['email_text']
        if email_text.strip() != '':
            result = classify_email(email_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)