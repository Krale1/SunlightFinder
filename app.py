from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import ast
from generate_dataset import generate_dataset
import os

app = Flask(__name__)

# Generate dataset if it doesn't exist
if not os.path.exists('cafes_dataset.csv'):
    generate_dataset('skopje_cafes.geojson')

def parse_dict_string(s):
    try:
        return ast.literal_eval(s)
    except:
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cafes')
def get_cafes():
    df = pd.read_csv('cafes_dataset.csv')
    
    # Convert string representations of dictionaries to actual dictionaries
    df['address'] = df['address'].apply(parse_dict_string)
    df['opening_hours'] = df['opening_hours'].apply(parse_dict_string)
    
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/cafes/<cafe_id>')
def get_cafe(cafe_id):
    df = pd.read_csv('cafes_dataset.csv')
    
    # Convert string representations of dictionaries to actual dictionaries
    df['address'] = df['address'].apply(parse_dict_string)
    df['opening_hours'] = df['opening_hours'].apply(parse_dict_string)
    
    cafe = df[df['id'] == cafe_id]
    if cafe.empty:
        return jsonify({'error': 'Cafe not found'}), 404
    return jsonify(cafe.iloc[0].to_dict())

@app.route('/cafes.geojson')
def cafes_geojson():
    return send_file('skopje_cafes.geojson', mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, port=5001) 