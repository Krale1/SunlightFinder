import pandas as pd
import numpy as np
import time
import os
import joblib
from datetime import datetime
import pytz
from pvlib.solarposition import get_solarposition
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import psutil
import warnings
import traceback
warnings.filterwarnings('ignore')

def compute_angle_diff(row):
    """Compute angle difference between sun azimuth and bar orientation."""
    try:
        lat, lon, hour, bar_angle = row['latitude'], row['longitude'], row['hour'], row['bar_orientation']
        if pd.isna(bar_angle):
            return 180  # Max difference if unknown

        tz = pytz.timezone('Europe/Skopje')
        now = datetime.now(tz).replace(hour=hour, minute=0, second=0, microsecond=0)
        sunpos = get_solarposition(now, lat, lon)
        sun_azimuth = sunpos['azimuth'].values[0]

        diff = abs(sun_azimuth - bar_angle)
        return min(diff, 360 - diff)
    except Exception as e:
        print(f"Error in compute_angle_diff: {str(e)}")
        print(f"Row data: {row}")
        return 180

def get_model_size(model, model_name):
    """Save model temporarily and get its size."""
    try:
        temp_path = f'temp_{model_name}.pkl'
        joblib.dump(model, temp_path)
        size = os.path.getsize(temp_path) / (1024 * 1024)  # Convert to MB
        os.remove(temp_path)
        return size
    except Exception as e:
        print(f"Error in get_model_size: {str(e)}")
        return 0.0

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return its performance metrics."""
    try:
        print(f"Training {model_name}...")
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f}s")

        print(f"Measuring prediction time for {model_name}...")
        # Prediction time (average over 100 predictions)
        start_time = time.time()
        for _ in range(100):
            model.predict(X_test[:1])
        prediction_time = (time.time() - start_time) / 100 * 1000  # Convert to ms
        print(f"Average prediction time: {prediction_time:.2f}ms")

        print(f"Calculating model size for {model_name}...")
        # Model size
        model_size = get_model_size(model, model_name)
        print(f"Model size: {model_size:.2f}MB")

        print(f"Computing performance metrics for {model_name}...")
        # Performance metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)
        print(f"Accuracy: {accuracy:.2%}")

        print(f"Running cross-validation for {model_name}...")
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"Cross-validation score: {cv_mean:.2%} (±{cv_std:.2%})")

        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            # Normalize feature importance for LightGBM
            if model_name == 'LightGBM':
                feature_importance = feature_importance / feature_importance.sum()
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0])
            # Normalize coefficients for linear models
            if model_name in ['Logistic Regression', 'SVM']:
                feature_importance = feature_importance / feature_importance.sum()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'model_size': model_size,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'feature_importance': feature_importance
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return None

def main():
    try:
        print("Loading and preprocessing data...")
        # Load dataset
        df = pd.read_csv('ML_dataset - cafes_dataset.csv')
        print(f"Dataset loaded successfully. Shape: {df.shape}")

        # Preprocessing
        print("Preprocessing data...")
        df['hour'] = df['hour'].str.extract(r'(\d+):')[0].astype(int)
        df['outdoor_seating'] = df['outdoor_seating'].astype(str).str.lower().map({'true': 1, 'false': 0})
        df['bar_orientation'] = pd.to_numeric(df['bar_orientation'], errors='coerce')
        print("Computing angle differences...")
        df['sun_bar_angle_diff'] = df.apply(compute_angle_diff, axis=1)
        df.dropna(subset=['sun_bar_angle_diff'], inplace=True)
        print(f"Preprocessing completed. Final shape: {df.shape}")

        # Features and target
        features = ['latitude', 'longitude', 'hour', 'outdoor_seating', 'sun_bar_angle_diff']
        X = df[features]
        y = df['is_in_sunlight'].astype(int)
        print(f"Features: {features}")
        print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        # Define models to test
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }

        # Evaluate each model
        results = {}
        print("\nEvaluating models...")
        for name, model in models.items():
            print(f"\n===== {name} =====")
            try:
                results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)
                if results[name] is not None:
                    print(f"✓ {name} evaluation completed successfully")
                else:
                    print(f"✗ {name} evaluation failed")
            except Exception as e:
                print(f"✗ Error evaluating {name}: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                results[name] = None

        # Generate report
        print("\nGenerating evaluation report...")
        with open('model_evaluation_results.txt', 'w') as f:
            f.write("Sunlight Finder - Model Evaluation Results (Real Numbers)\n")
            f.write("====================================================\n\n")
            
            f.write("Testing Methodology\n")
            f.write("------------------\n")
            f.write("1. Dataset Preparation:\n")
            f.write(f"   - Dataset size: {len(df)} samples\n")
            f.write(f"   - Features: {', '.join(features)}\n")
            f.write(f"   - Target distribution: {y.value_counts(normalize=True).to_dict()}\n")
            f.write("   - Train/Test split: 75%/25%\n")
            f.write("   - Cross-validation: 5-fold\n\n")

            f.write("Model Testing Results\n")
            f.write("-------------------\n\n")
            
            for name, result in results.items():
                if result is None:
                    f.write(f"{name}\n")
                    f.write("Evaluation failed\n\n")
                    continue
                    
                f.write(f"{name}\n")
                f.write(f"Accuracy: {result['accuracy']:.1%}\n")
                f.write(f"Precision (Sun/No Sun): {result['precision'][1]:.1%}/{result['precision'][0]:.1%}\n")
                f.write(f"Recall (Sun/No Sun): {result['recall'][1]:.1%}/{result['recall'][0]:.1%}\n")
                f.write(f"F1-score (Sun/No Sun): {result['f1'][1]:.1%}/{result['f1'][0]:.1%}\n")
                f.write(f"Avg. Prediction Time: {result['prediction_time']:.2f}ms\n")
                f.write(f"Model Size: {result['model_size']:.1f}MB\n")
                f.write(f"Training Time: {result['training_time']:.1f}s\n")
                f.write(f"Cross-validation Score: {result['cv_mean']:.1%} (±{result['cv_std']:.1%})\n")
                
                if result['feature_importance'] is not None:
                    f.write("\nFeature Importance:\n")
                    importance_dict = dict(zip(features, result['feature_importance']))
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    for feature, importance in sorted_importance:
                        f.write(f"   {feature}: {importance:.1%}\n")
                f.write("\n")

            # Add analysis and recommendations
            f.write("\nModel Selection Analysis\n")
            f.write("----------------------\n")
            
            # Find best model by accuracy
            valid_results = {k: v for k, v in results.items() if v is not None}
            if valid_results:
                best_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
                f.write(f"\nBest performing model: {best_model[0]} ({best_model[1]['accuracy']:.1%} accuracy)\n")
                
                f.write("\nModel Comparison:\n")
                for name, result in valid_results.items():
                    f.write(f"\n{name}:\n")
                    f.write(f"  - Accuracy: {result['accuracy']:.1%}\n")
                    f.write(f"  - Prediction Time: {result['prediction_time']:.2f}ms\n")
                    f.write(f"  - Model Size: {result['model_size']:.1f}MB\n")
                    f.write(f"  - Training Time: {result['training_time']:.1f}s\n")
                
                f.write("\nRecommendations:\n")
                f.write("1. Consider the trade-off between accuracy and computational efficiency\n")
                f.write("2. Evaluate model performance in real-world conditions\n")
                f.write("3. Consider model interpretability and maintenance requirements\n")
                f.write("4. Test model performance with different feature combinations\n")
            else:
                f.write("\nNo models were successfully evaluated.\n")

        print("\nEvaluation complete! Results saved to 'model_evaluation_results.txt'")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 