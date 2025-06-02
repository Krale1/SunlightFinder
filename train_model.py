import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from pvlib.solarposition import get_solarposition
from datetime import datetime
import pytz

# Load dataset
df = pd.read_csv('ML_dataset - cafes_dataset.csv')

# Preprocessing
df['hour'] = df['hour'].str.extract(r'(\d+):')[0].astype(int)
df['outdoor_seating'] = df['outdoor_seating'].astype(str).str.lower().map({'true': 1, 'false': 0})

# Clean bar_orientation and convert to float
df['bar_orientation'] = pd.to_numeric(df['bar_orientation'], errors='coerce')

# Define a function to compute angle difference between sun azimuth and bar orientation
def compute_angle_diff(row):
    lat, lon, hour, bar_angle = row['latitude'], row['longitude'], row['hour'], row['bar_orientation']
    if pd.isna(bar_angle):
        return 180  # Max difference if unknown

    tz = pytz.timezone('Europe/Skopje')
    now = datetime.now(tz).replace(hour=hour, minute=0, second=0, microsecond=0)
    sunpos = get_solarposition(now, lat, lon)
    sun_azimuth = sunpos['azimuth'].values[0]

    diff = abs(sun_azimuth - bar_angle)
    return min(diff, 360 - diff)

# Compute angle difference
df['sun_bar_angle_diff'] = df.apply(compute_angle_diff, axis=1)

# Drop rows with missing values
df.dropna(subset=['sun_bar_angle_diff'], inplace=True)

# Features and target
features = ['latitude', 'longitude', 'hour', 'outdoor_seating', 'sun_bar_angle_diff']
X = df[features]
y = df['is_in_sunlight'].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("\n✅ Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model
joblib.dump(model, 'sunlight_model_with_angle.pkl')
print("\n✅ Model saved as 'sunlight_model_with_angle.pkl'")
