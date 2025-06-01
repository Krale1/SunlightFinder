from flask import Flask, render_template, jsonify, send_file, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load ML model and encoder
model = joblib.load('sunlight_model.pkl')
orientation_encoder = joblib.load('orientation_encoder.pkl')

# Define feature names to match training
FEATURE_NAMES = ['latitude', 'longitude', 'hour', 'outdoor_seating', 'bar_orientation_encoded']

def get_direction(angle):
    """Convert angle to cardinal direction."""
    if pd.isna(angle):
        return 'unknown'
    # Convert angle to 8 cardinal directions
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = round(angle / 45) % 8
    return directions[index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cafes.geojson')
def cafes_geojson():
    """Serve the GeoJSON file with cafe data."""
    return send_file('skopje_cafes.geojson', mimetype='application/json')

@app.route('/predict', methods=['GET'])
def predict():
    """Single cafe prediction endpoint using ML model."""
    try:
        print("[DEBUG] Request args:", request.args)

        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        bar_orientation = request.args.get('orientation')

        hour_raw = request.args.get('hour', '').strip()
        if not hour_raw.isdigit():
            raise ValueError(f"Invalid hour: {hour_raw}")
        hour = int(hour_raw)

        outdoor_raw = request.args.get('outdoor', '0').lower().strip()
        outdoor = 1 if outdoor_raw in ['true', '1', 'yes'] else 0

        # Convert bar orientation to encoded value
        if bar_orientation is not None:
            try:
                bar_orientation = float(bar_orientation)
                direction = get_direction(bar_orientation)
                orientation_encoded = orientation_encoder.transform([direction])[0]
            except (ValueError, TypeError):
                # If invalid orientation, use 'unknown'
                orientation_encoded = orientation_encoder.transform(['unknown'])[0]
        else:
            orientation_encoded = orientation_encoder.transform(['unknown'])[0]

        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([[lat, lng, hour, outdoor, orientation_encoded]], 
                                 columns=FEATURE_NAMES)
        prediction = model.predict(features_df)[0]
        emoji = '☀️' if prediction == 1 else '🌑'

        return jsonify({'emoji': emoji, 'sun': bool(prediction)})
    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple cafes using ML model."""
    try:
        data = request.get_json()
        hour = int(data['hour'])
        cafes = data['cafes']
        
        # Prepare features for all cafes
        features_list = []
        for cafe in cafes:
            lat = float(cafe['lat'])
            lng = float(cafe['lng'])
            outdoor_raw = str(cafe.get('outdoor', 'false')).lower()
            outdoor = 1 if outdoor_raw in ['true', '1', 'yes'] else 0
            bar_orientation = cafe.get('orientation')

            # Convert bar orientation to encoded value
            if bar_orientation is not None:
                try:
                    bar_orientation = float(bar_orientation)
                    direction = get_direction(bar_orientation)
                    orientation_encoded = orientation_encoder.transform([direction])[0]
                except (ValueError, TypeError):
                    # If invalid orientation, use 'unknown'
                    orientation_encoded = orientation_encoder.transform(['unknown'])[0]
            else:
                orientation_encoded = orientation_encoder.transform(['unknown'])[0]

            features_list.append([lat, lng, hour, outdoor, orientation_encoded])

        # Create DataFrame with proper feature names
        features_df = pd.DataFrame(features_list, columns=FEATURE_NAMES)
        predictions = model.predict(features_df)
        
        results = []
        for prediction in predictions:
            emoji = '☀️' if prediction == 1 else '🌑'
            results.append({
                'emoji': emoji,
                'sun': bool(prediction)
            })

        return jsonify(results)
    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
