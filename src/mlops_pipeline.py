import os
import sys

def run_pipeline():
    print("Iniciando Pipeline de Sentiment Analysis...")

    # Defining steps in logical order
    # 1. Cleansing/Preprocessing -> 2. Training -> 3. Evaluation/Visualization
    steps = [
        ("Preprocessing", "python src/preprocessing.py"),
        ("Training SVM (Optimized)", "python src/model_training.py"),
        ("Evaluation y Visualization", "python src/evaluation.py")
    ]

    for step_name, command in steps:
        print(f"\n--- Executing: {step_name} ---")
        
        # Execute code in terminal
        exit_code = os.system(command)
        
        # If exit code is not 0, there was an error
        if exit_code != 0:
            print(f"ERROR: In step '{step_name}' failed.")
            print("Pipeline has stopped tp avoid an error cascade.")
            sys.exit(1)

    print("\n" + "="*40)
    print("PIPELINE FINALIZED SUCCESSFULLY")
    print("Metrics saved in MLflow and reports in folder /data")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()