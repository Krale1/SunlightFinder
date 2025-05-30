import json
import pandas as pd
from datetime import datetime

def get_all_hours():
    """Generate a list of all hours in 24-hour format."""
    return [f"{hour:02d}:00" for hour in range(24)]

def parse_opening_hours(hours_str):
    """Parse opening hours string into a more structured format."""
    if not hours_str:
        return None
    
    try:
        # Split into different day ranges
        day_ranges = hours_str.split(';')
        parsed_hours = {}
        
        for day_range in day_ranges:
            days, hours = day_range.split(' ', 1)
            parsed_hours[days] = hours
            
        return parsed_hours
    except:
        return None

def generate_dataset(geojson_file, output_file='cafes_dataset.csv'):
    """Generate a structured dataset from the GeoJSON file."""
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cafes = []
    all_hours = get_all_hours()
    
    for feature in data['features']:
        props = feature['properties']
        geometry = feature['geometry']
        
        # Create a base cafe entry
        base_cafe = {
            'name': props.get('name', ''),
            'latitude': geometry['coordinates'][1],
            'longitude': geometry['coordinates'][0],
            'opening_hours': parse_opening_hours(props.get('opening_hours')),
            'outdoor_seating': props.get('outdoor_seating', 'no') == 'yes',
            # 'indoor': props.get('indoor', 0) == 1,
            'bar_orientation': props.get('bar_orientation', None),
        }
        
        # Create a row for each hour
        for hour in all_hours:
            cafe_entry = base_cafe.copy()
            cafe_entry['hour'] = hour
            cafe_entry['is_in_sunlight'] = 0  # Default value, to be updated later
            cafes.append(cafe_entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(cafes)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Dataset generated successfully: {output_file}")
    print(f"Total rows generated: {len(cafes)}")
    print(f"Total unique cafes: {len(data['features'])}")
    
    return df

if __name__ == "__main__":
    generate_dataset('skopje_cafes.geojson') 