from flask import Flask, render_template, jsonify, send_file, request
import pandas as pd
import joblib
from datetime import datetime
from pvlib.solarposition import get_solarposition
import pytz

app = Flask(__name__)

# Load ML model
model = joblib.load('sunlight_model_with_angle.pkl')

# Define feature names for the model
FEATURE_NAMES = ['latitude', 'longitude', 'hour', 'outdoor_seating', 'sun_bar_angle_diff']


def compute_angle_diff(lat, lng, hour, bar_angle):
    if bar_angle is None:
        return 180  # Default value if no orientation is given

    try:
        tz = pytz.timezone('Europe/Skopje')
        now = datetime.now(tz).replace(hour=hour, minute=0, second=0, microsecond=0)
        sunpos = get_solarposition(now, lat, lng)
        sun_azimuth = sunpos['azimuth'].values[0]

        diff = abs(sun_azimuth - bar_angle)
        return min(diff, 360 - diff)
    except:
        return 180


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cafes.geojson')
def cafes_geojson():
    return send_file('skopje_cafes.geojson', mimetype='application/json')


@app.route('/predict', methods=['GET'])
def predict():
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        hour = int(request.args.get('hour'))
        outdoor_raw = request.args.get('outdoor', '0').lower().strip()
        outdoor = 1 if outdoor_raw in ['true', '1', 'yes'] else 0

        orientation = request.args.get('orientation')
        try:
            orientation = float(orientation)
        except:
            orientation = None

        angle_diff = compute_angle_diff(lat, lng, hour, orientation)

        features_df = pd.DataFrame([[lat, lng, hour, outdoor, angle_diff]], columns=FEATURE_NAMES)
        prediction = model.predict(features_df)[0]
        emoji = '☀️' if prediction == 1 else '🌑'

        return jsonify({'emoji': emoji, 'sun': bool(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        hour = int(data['hour'])
        cafes = data['cafes']

        features_list = []
        for cafe in cafes:
            lat = float(cafe['lat'])
            lng = float(cafe['lng'])
            outdoor_raw = str(cafe.get('outdoor', 'false')).lower()
            outdoor = 1 if outdoor_raw in ['true', '1', 'yes'] else 0

            orientation = cafe.get('orientation')
            try:
                orientation = float(orientation)
            except:
                orientation = None

            angle_diff = compute_angle_diff(lat, lng, hour, orientation)

            features_list.append([lat, lng, hour, outdoor, angle_diff])

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
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)
