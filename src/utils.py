# src/utils.py
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import streamlit as st

def calculate_speed_kmh(distance_km, time_hours):
    """Calculate speed in km/h, handling edge cases"""
    if time_hours <= 0 or distance_km <= 0:
        return 0
    return distance_km / time_hours

def is_physically_possible(distance_km, time_hours):
    """
    Check if delivery is physically possible
    Returns: (is_possible, speed, category)
    """
    if time_hours <= 0 or distance_km <= 0:
        return False, 0, "invalid"
    
    speed = calculate_speed_kmh(distance_km, time_hours)
    
    # Physical limits
    if speed < 3:    # Slower than walking pace
        return False, speed, "too_slow"
    if speed > 900:  # Faster than commercial planes
        return False, speed, "too_fast"
    
    # Categorize realistic speeds
    if speed <= 20:
        return True, speed, "critical_slow"
    elif speed <= 40:
        return True, speed, "slow"
    elif speed <= 120:
        return True, speed, "normal"
    else:
        return True, speed, "fast"

def categorize_delivery_speed(speed):
    """Categorize speed for visualization"""
    if speed <= 20:
        return "Critical Slow (3-20 km/h)"
    elif speed <= 40:
        return "Slow (20-40 km/h)"
    elif speed <= 120:
        return "Normal (40-120 km/h)"
    else:
        return "Fast (120-900 km/h)"

def detect_operational_anomalies(distance_km, time_hours, speed_kmh):
    """
    Detect operational anomalies (slow but physically possible)
    """
    if time_hours <= 0:
        return "invalid_time"
    
    # Operational problem detection
    if speed_kmh < 20 and time_hours > 2:
        return "critical_delay"
    elif speed_kmh < 40 and time_hours > 4:
        return "major_delay"
    elif speed_kmh < 60 and time_hours > 8:
        return "moderate_delay"
    elif time_hours > 24:
        return "extended_delay"
    else:
        return "normal"

def validate_prediction_inputs(distance, city, service_type, hour, day_of_week):
    """
    Validate prediction inputs and return errors or warnings
    """
    errors = []
    warnings = []
    
    # Distance validation
    if distance <= 0:
        errors.append("Distance must be greater than 0")
    elif distance > 1000:
        warnings.append("Distance exceeds 1000km - verify this is correct")
    elif distance < 1:
        warnings.append("Distance is very short - is this within the same city?")
    
    # Time validation
    if not (0 <= hour <= 23):
        errors.append("Hour must be between 0 and 23")
    
    # Service type validation
    valid_services = ["EXPRESS", "STANDARD", "ECONOMY", "PREMIUM"]
    if service_type not in valid_services:
        warnings.append(f"Service type '{service_type}' may not be recognized by the model")
    
    return errors, warnings

def apply_prediction_sanity_checks(prediction, distance, service_type, hour, day_of_week):
    """
    Apply sanity checks to predictions and adjust if needed
    Returns: (adjusted_prediction, warnings)
    """
    warnings = []
    adjusted_prediction = prediction
    
    # Calculate reasonable bounds based on distance
    min_speed = 3  # Minimum realistic speed (walking pace)
    max_speed = 120  # Maximum realistic speed (highway driving)
    
    min_time = distance / max_speed  # Fastest possible time
    max_time = distance / min_speed  # Slowest reasonable time
    
    # Adjust for service type
    service_factors = {
        "EXPRESS": 0.7,   # 30% faster than standard
        "STANDARD": 1.0,  # Baseline
        "ECONOMY": 1.3,   # 30% slower than standard
        "PREMIUM": 0.6,   # 40% faster than standard
    }
    
    service_factor = service_factors.get(service_type, 1.0)
    min_time *= service_factor
    max_time *= service_factor
    
    # Adjust for time of day (night deliveries might be slower)
    if 22 <= hour <= 6:  # Night hours
        min_time *= 1.2  # 20% slower at night
        max_time *= 1.2
    
    # Adjust for weekends (might be faster due to less traffic)
    if day_of_week in ["Saturday", "Sunday"]:
        min_time *= 0.9  # 10% faster on weekends
        max_time *= 0.9
    
    # Apply sanity checks
    if prediction < min_time:
        warnings.append(f"Prediction ({prediction:.2f}h) seems too fast for {distance}km with {service_type} service")
        adjusted_prediction = min_time * 1.1  # Add 10% buffer to minimum
        warnings.append(f"Adjusted to {adjusted_prediction:.2f}h based on physical limits")
        
    elif prediction > max_time:
        warnings.append(f"Prediction ({prediction:.2f}h) seems too slow for {distance}km with {service_type} service")
        adjusted_prediction = max_time * 0.9  # Use 90% of maximum
        warnings.append(f"Adjusted to {adjusted_prediction:.2f}h based on physical limits")
    
    # Check for extremely short deliveries
    if distance > 10 and adjusted_prediction < 0.5:
        warnings.append(f"Delivery time seems too short for {distance}km distance")
        adjusted_prediction = max(adjusted_prediction, 0.5)  # At least 30 minutes
    
    # Check for extremely long deliveries
    if adjusted_prediction > 72:  # More than 3 days
        warnings.append(f"Delivery time seems excessively long for {distance}km distance")
    
    return adjusted_prediction, warnings

def calculate_realistic_time_bounds(distance, service_type, hour, day_of_week):
    """
    Calculate realistic time bounds for a given delivery scenario
    Returns: (min_time, max_time, expected_time)
    """
    # Base speeds (km/h)
    base_speeds = {
        "EXPRESS": 60,
        "STANDARD": 40,
        "ECONOMY": 25,
        "PREMIUM": 70,
    }
    
    base_speed = base_speeds.get(service_type, 40)
    
    # Adjust for time of day
    if 22 <= hour <= 6:  # Night hours
        speed_factor = 0.8  # 20% slower at night
    elif 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
        speed_factor = 0.7  # 30% slower during rush hour
    else:
        speed_factor = 1.0  # Normal speed
    
    # Adjust for weekends
    if day_of_week in ["Saturday", "Sunday"]:
        speed_factor *= 1.1  # 10% faster on weekends (less traffic)
    
    adjusted_speed = base_speed * speed_factor
    
    # Calculate times
    min_speed = adjusted_speed * 1.3  # Maximum possible speed (30% faster than adjusted)
    max_speed = adjusted_speed * 0.7  # Minimum reasonable speed (30% slower than adjusted)
    
    min_time = distance / min_speed
    max_time = distance / max_speed
    expected_time = distance / adjusted_speed
    
    return min_time, max_time, expected_time

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic delivery data for demonstration purposes
    """
    np.random.seed(42)
    
    # Cities in Tanzania with approximate coordinates
    cities = {
        "Dar es Salaam": (-6.7924, 39.2083),
        "Arusha": (-3.3869, 36.6830),
        "Mwanza": (-2.5167, 32.9000),
        "Mbeya": (-8.9000, 33.4500),
        "Dodoma": (-6.1630, 35.7516)
    }
    
    # Service types
    service_types = ["EXPRESS", "STANDARD", "ECONOMY"]
    
    # Generate data
    data = {
        "order_id": [f"ORD_{i:06d}" for i in range(n_samples)],
        "from_city_name": np.random.choice(list(cities.keys()), n_samples),
        "to_city_name": np.random.choice(list(cities.keys()), n_samples),
        "typecode": np.random.choice(service_types, n_samples),
        "distance_km": np.random.uniform(1, 500, n_samples).round(2),
        "receipt_time": [datetime(2024, 1, 1) + timedelta(
            days=np.random.randint(1, 365),
            hours=np.random.randint(0, 24)
        ) for _ in range(n_samples)],
    }
    
    df = pd.DataFrame(data)
    
    # Add coordinates based on cities
    df['poi_lat'] = df['from_city_name'].map(lambda x: cities[x][0])
    df['poi_lng'] = df['from_city_name'].map(lambda x: cities[x][1])
    df['receipt_lat'] = df['to_city_name'].map(lambda x: cities[x][0])
    df['receipt_lng'] = df['to_city_name'].map(lambda x: cities[x][1])
    
    # Calculate realistic delivery times based on distance and service type
    def calculate_delivery_time(row):
        base_time = row['distance_km'] / 40  # Base speed 40 km/h
        
        # Service type modifiers
        if row['typecode'] == "EXPRESS":
            base_time *= 0.7  # 30% faster
        elif row['typecode'] == "ECONOMY":
            base_time *= 1.3  # 30% slower
        
        # Add some randomness
        base_time *= np.random.uniform(0.8, 1.2)
        
        return max(0.5, base_time)  # Minimum 0.5 hours
    
    df['delivery_time_hours'] = df.apply(calculate_delivery_time, axis=1)
    df['sign_time'] = df['receipt_time'] + pd.to_timedelta(df['delivery_time_hours'], 'hours')
    
    # Add some weather data
    df['temperature'] = np.random.uniform(20, 35, n_samples).round(1)
    df['precipitation'] = np.random.exponential(2, n_samples).round(1)
    df['windspeed'] = np.random.uniform(0, 20, n_samples).round(1)
    
    return df

def format_delivery_time(hours):
    """Format delivery time in human-readable format"""
    if hours < 1:
        return f"{hours*60:.0f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"

def calculate_estimated_arrival(pickup_time, predicted_hours):
    """Calculate estimated arrival datetime"""
    if hasattr(pickup_time, 'tzinfo'):
        return pickup_time + timedelta(hours=predicted_hours)
    else:
        return pickup_time + pd.Timedelta(hours=predicted_hours)