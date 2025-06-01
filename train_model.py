import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('ML_dataset - cafes_dataset.csv')

# Preprocessing
df['hour'] = df['hour'].str.extract(r'(\d+):')[0].astype(int)
df['outdoor_seating'] = df['outdoor_seating'].astype(str).str.lower().map({'true': 1, 'false': 0})

# Handle bar orientation - convert to categorical bins (N, NE, E, SE, S, SW, W, NW)
def get_direction(angle):
    if pd.isna(angle):
        return 'unknown'
    # Convert angle to 8 cardinal directions
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = round(angle / 45) % 8
    return directions[index]

df['bar_orientation'] = df['bar_orientation'].apply(get_direction)

# Encode categorical features
orientation_encoder = LabelEncoder()
df['bar_orientation_encoded'] = orientation_encoder.fit_transform(df['bar_orientation'])

# Save the encoder for later use
joblib.dump(orientation_encoder, 'orientation_encoder.pkl')

features = ['latitude', 'longitude', 'hour', 'outdoor_seating', 'bar_orientation_encoded']
X = df[features]
y = df['is_in_sunlight'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

joblib.dump(model, 'sunlight_model.pkl')
print("\n✅ Model saved as 'sunlight_model.pkl'")
