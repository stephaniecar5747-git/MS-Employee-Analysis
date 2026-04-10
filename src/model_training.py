import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm():
    # Load the preprocessed data
    df = pd.read_csv("data/processed_en.csv")
    
    
    # Define Target: Positive (1) if overall_ratings > 3, else Negative (0)
    df['label'] = (df['overall-ratings'] > 3).astype(int)
    
    # SVM requires strings, so we join the list of lemmas back into a sentence
    # If your preprocessing saved them as strings, this step is simple:
    df = df.dropna(subset=['full_review_lemmas'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['full_review_lemmas'], df['label'], test_size=0.3, random_state=42
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_df=0.8, 
        min_df=5,
        max_features=5000
        )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    mlflow.set_experiment("Employee_Sentiment_Analysis")
    
    with mlflow.start_run(run_name="SVM_Model_Final"):
        # Task: Construction model using SVM
        # probability=True is required for "Calculation of grammatical probabilities"
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train_vec, y_train)
        
        # Calculation of grammatical probabilities (confidence scores)
        sample_probs = model.predict_proba(X_test_vec[:5])
        
        # Logging to MLOps (MLflow)
        acc = accuracy_score(y_test, model.predict(X_test_vec))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "svm_sentiment_model")
        
        print(f"Model Training Complete. Accuracy: {acc:.2%}")
        return model, vectorizer

if __name__ == "__main__":
    train_svm()