import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    df = pd.read_csv('spam.csv')
    df = df.copy().drop_duplicates(keep='first')
    df['Label'] = df['Label'].map({'spam': 1, 'ham': 0})
    
    x = df['EmailText'].fillna('')
    y = df['Label']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    tfidf1 = TfidfVectorizer()
    feature = tfidf1.fit_transform(x_train)
    
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(feature, y_train)
    
    return svm_model, tfidf1