import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_svm():
    # 1. Load data
    df = pd.read_csv("data/processed_en.csv")
    
    # Column name management (reviewing compatibility)
    col_rating = 'overall-ratings' if 'overall-ratings' in df.columns else 'overall_ratings'
    df['label'] = (df[col_rating] > 3).astype(int)
    
    df = df.dropna(subset=['full_review_lemmas'])
    
    # 2. Data split
    X_train, X_test, y_train, y_test = train_test_split(
        df['full_review_lemmas'], df['label'], test_size=0.3, random_state=42
    )

    # 3. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_df=0.8, 
        min_df=5,
        max_features=5000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    mlflow.set_experiment("Employee_Sentiment_Analysis")
    
    with mlflow.start_run(run_name="SVM_GridSearch_Optimization"):
        
        # --- Definition of parameter search ---
        param_grid = {
            'C': [0.1, 1, 10],              # Error penalty
            'kernel': ['linear', 'rbf'],    # separation type
            'class_weight': ['balanced', None] # Class imbalance management
        }

        print("🔍 Init GridSearchCV (this can take a moment)...")
        # cv=3 cross validation in 3 parts
        grid_search = GridSearchCV(
            SVC(probability=True), 
            param_grid, 
            refit=True, 
            verbose=2, 
            cv=3,
            n_jobs=-1 # Uses all CPU nucleus
        )
        
        grid_search.fit(X_train_vec, y_train)
        
        # 4. Select the best model
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters found: {grid_search.best_params_}")

        # 5. Metric evaluation
        y_pred = best_model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        
        # Registration in ML flow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(best_model, "svm_sentiment_model")
        
        # Probabilities calculation
        sample_probs = best_model.predict_proba(X_test_vec[:5])
        
        print(f"Optimized training complete. Precision: {acc:.2%}")
        print("\nClassification detail:")
        print(classification_report(y_test, y_pred))
        
        return best_model, vectorizer

if __name__ == "__main__":
    train_svm()