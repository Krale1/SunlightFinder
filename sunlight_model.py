import math
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple

class SunlightModel:
    def __init__(self):
        # Skopje coordinates
        self.latitude = 41.998
        self.longitude = 21.435
        
    def calculate_sun_position(self, date: datetime) -> Tuple[float, float]:
        """
        Calculate sun's azimuth and elevation for a given date and time.
        Returns (azimuth, elevation) in degrees.
        """
        # Convert date to Julian day
        jd = self._to_julian_day(date)
        
        # Calculate solar position
        n = jd - 2451545.0  # Days since J2000.0
        
        # Mean longitude of the sun
        L = 280.460 + 0.9856474 * n
        
        # Mean anomaly of the sun
        g = 357.528 + 0.9856003 * n
        
        # Ecliptic longitude
        lambda_sun = L + 1.915 * math.sin(math.radians(g)) + 0.020 * math.sin(math.radians(2 * g))
        
        # Obliquity of the ecliptic
        epsilon = 23.439 - 0.0000004 * n
        
        # Convert to radians
        lambda_sun_rad = math.radians(lambda_sun)
        epsilon_rad = math.radians(epsilon)
        
        # Calculate right ascension and declination
        alpha = math.atan2(
            math.cos(epsilon_rad) * math.sin(lambda_sun_rad),
            math.cos(lambda_sun_rad)
        )
        delta = math.asin(math.sin(epsilon_rad) * math.sin(lambda_sun_rad))
        
        # Calculate hour angle
        lst = self._local_sidereal_time(date)
        h = lst - math.degrees(alpha)
        
        # Convert to radians
        h_rad = math.radians(h)
        lat_rad = math.radians(self.latitude)
        
        # Calculate elevation and azimuth
        elevation = math.asin(
            math.sin(lat_rad) * math.sin(delta) +
            math.cos(lat_rad) * math.cos(delta) * math.cos(h_rad)
        )
        
        azimuth = math.atan2(
            math.sin(h_rad),
            math.cos(h_rad) * math.sin(lat_rad) - math.tan(delta) * math.cos(lat_rad)
        )
        
        # Convert to degrees
        elevation = math.degrees(elevation)
        azimuth = math.degrees(azimuth)
        
        # Normalize azimuth to 0-360
        azimuth = (azimuth + 360) % 360
        
        return azimuth, elevation
    
    def _to_julian_day(self, date: datetime) -> float:
        """Convert datetime to Julian day."""
        year = date.year
        month = date.month
        day = date.day
        hour = date.hour + date.minute / 60.0 + date.second / 3600.0
        
        if month <= 2:
            year -= 1
            month += 12
            
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd += hour / 24.0
        
        return jd
    
    def _local_sidereal_time(self, date: datetime) -> float:
        """Calculate Local Sidereal Time."""
        jd = self._to_julian_day(date)
        t = (jd - 2451545.0) / 36525.0
        
        # Greenwich Mean Sidereal Time
        gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
               t * t * (0.000387933 - t / 38710000.0)
        
        # Local Sidereal Time
        lst = gmst + self.longitude
        
        return lst % 360
    
    def predict_sunlight(self, 
                        bar_orientation: float, 
                        date: datetime, 
                        cloud_coverage: float) -> bool:
        """
        Predict if a cafe is in sunlight based on:
        - bar_orientation: orientation of the bar in degrees (0-360)
        - date: datetime object for the current time
        - cloud_coverage: cloud coverage percentage (0-100)
        
        Returns True if the cafe is in sunlight, False otherwise.
        """
        # Get sun position
        sun_azimuth, sun_elevation = self.calculate_sun_position(date)
        
        # If sun is below horizon, no sunlight
        if sun_elevation < 0:
            return False
        
        # Calculate angle difference between sun and bar orientation
        angle_diff = abs(sun_azimuth - bar_orientation)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # Calculate sunlight probability based on angle difference
        # Maximum sunlight when angle difference is 0 or 180 degrees
        angle_factor = math.cos(math.radians(angle_diff))
        
        # Calculate elevation factor
        # More sunlight when sun is higher in the sky
        elevation_factor = math.sin(math.radians(sun_elevation))
        
        # Calculate cloud factor
        # Less sunlight with more cloud coverage
        cloud_factor = 1 - (cloud_coverage / 100)
        
        # Combine factors
        sunlight_probability = angle_factor * elevation_factor * cloud_factor
        
        # Threshold for determining if in sunlight
        return sunlight_probability > 0.3
    
    def update_cafe_sunlight_status(self, cafes_df: pd.DataFrame, 
                                  date: datetime, 
                                  cloud_coverage: float) -> pd.DataFrame:
        """
        Update the is_in_sunlight status for all cafes in the dataframe.
        
        Args:
            cafes_df: DataFrame containing cafe data with bar_orientation column
            date: datetime object for the current time
            cloud_coverage: cloud coverage percentage (0-100)
            
        Returns:
            Updated DataFrame with is_in_sunlight column
        """
        # Create a copy of the dataframe
        updated_df = cafes_df.copy()
        
        # Update sunlight status for each cafe
        updated_df['is_in_sunlight'] = updated_df['bar_orientation'].apply(
            lambda orientation: self.predict_sunlight(orientation, date, cloud_coverage)
        )
        
        return updated_df

# Example usage:
if __name__ == "__main__":
    # Create model instance
    model = SunlightModel()
    
    # Example date and cloud coverage
    current_date = datetime.now()
    cloud_coverage = 20  # 20% cloud coverage
    
    # Example bar orientation
    bar_orientation = 45  # 45 degrees
    
    # Predict sunlight
    is_sunny = model.predict_sunlight(bar_orientation, current_date, cloud_coverage)
    print(f"Cafe is {'in sunlight' if is_sunny else 'in shadow'}") 