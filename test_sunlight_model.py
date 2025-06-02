import pandas as pd
from datetime import datetime, timedelta
from sunlight_model import SunlightModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def create_test_dataset():
    """Create a test dataset with known sunlight conditions."""
    # Create test cases with different orientations and times
    test_cases = []
    
    # Test different orientations (0, 45, 90, 135, 180, 225, 270, 315 degrees)
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # Test different times of day (every 2 hours)
    times = [datetime.now().replace(hour=h, minute=0, second=0, microsecond=0) 
            for h in range(6, 20, 2)]
    
    # Test different cloud coverage levels
    cloud_coverage_levels = [0, 25, 50, 75, 100]
    
    # Generate test cases
    for orientation in orientations:
        for time in times:
            for cloud_coverage in cloud_coverage_levels:
                test_cases.append({
                    'bar_orientation': orientation,
                    'time': time,
                    'cloud_coverage': cloud_coverage
                })
    
    return pd.DataFrame(test_cases)

def get_actual_sunlight_status(row, model):
    """Get actual sunlight status based on sun position and orientation."""
    sun_azimuth, sun_elevation = model.calculate_sun_position(row['time'])
    
    # If sun is below horizon, no sunlight
    if sun_elevation < 0:
        return False
    
    # Calculate angle difference between sun and bar orientation
    angle_diff = abs(sun_azimuth - row['bar_orientation'])
    angle_diff = min(angle_diff, 360 - angle_diff)
    
    # Consider a cafe to be in sunlight if:
    # 1. Sun is above horizon
    # 2. Angle difference is less than 45 degrees (facing sun)
    # 3. Cloud coverage is less than 50%
    return (angle_diff < 45 and 
            sun_elevation > 0 and 
            row['cloud_coverage'] < 50)

def evaluate_model():
    """Evaluate the model's accuracy using test data."""
    # Initialize model
    model = SunlightModel()
    
    # Create test dataset
    test_df = create_test_dataset()
    
    # Get actual sunlight status
    test_df['actual_sunlight'] = test_df.apply(
        lambda row: get_actual_sunlight_status(row, model), axis=1
    )
    
    # Get model predictions
    test_df['predicted_sunlight'] = test_df.apply(
        lambda row: model.predict_sunlight(
            row['bar_orientation'],
            row['time'],
            row['cloud_coverage']
        ), axis=1
    )
    
    # Calculate metrics
    accuracy = accuracy_score(test_df['actual_sunlight'], test_df['predicted_sunlight'])
    precision = precision_score(test_df['actual_sunlight'], test_df['predicted_sunlight'])
    recall = recall_score(test_df['actual_sunlight'], test_df['predicted_sunlight'])
    f1 = f1_score(test_df['actual_sunlight'], test_df['predicted_sunlight'])
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        test_df['actual_sunlight'],
        test_df['predicted_sunlight'],
        rownames=['Actual'],
        colnames=['Predicted']
    )
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    # Plot accuracy by time of day
    plt.figure(figsize=(10, 6))
    time_accuracy = test_df.groupby(test_df['time'].dt.hour).apply(
        lambda x: accuracy_score(x['actual_sunlight'], x['predicted_sunlight'])
    )
    time_accuracy.plot(kind='bar')
    plt.title('Model Accuracy by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_accuracy_by_hour.png')
    
    # Plot accuracy by orientation
    plt.figure(figsize=(10, 6))
    orientation_accuracy = test_df.groupby('bar_orientation').apply(
        lambda x: accuracy_score(x['actual_sunlight'], x['predicted_sunlight'])
    )
    orientation_accuracy.plot(kind='bar')
    plt.title('Model Accuracy by Bar Orientation')
    plt.xlabel('Bar Orientation (degrees)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_accuracy_by_orientation.png')
    
    # Plot accuracy by cloud coverage
    plt.figure(figsize=(10, 6))
    cloud_accuracy = test_df.groupby('cloud_coverage').apply(
        lambda x: accuracy_score(x['actual_sunlight'], x['predicted_sunlight'])
    )
    cloud_accuracy.plot(kind='bar')
    plt.title('Model Accuracy by Cloud Coverage')
    plt.xlabel('Cloud Coverage (%)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_accuracy_by_clouds.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix
    }

if __name__ == "__main__":
    results = evaluate_model() 