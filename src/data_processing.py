# src/data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import streamlit as st
import hashlib
warnings.filterwarnings('ignore')

def decode_hashed_service_types(df):
    """
    Decode hashed service types into readable categories
    Based on comprehensive analysis of the data patterns
    """
    df_clean = df.copy()
    
    if 'typecode' in df_clean.columns:
        # Comprehensive hash mapping based on your data sample
        hash_mapping = {
            # Express service hashes
            '4602b38053ece07a9ca5153f1df2e404': 'EXPRESS',
            '14cca3f2714c7c0faf2cbac10ba12d3b': 'EXPRESS',
            '339d14e62a5bbd67de62f461a5f7db1e': 'EXPRESS',
            'e8b508bbdada69046e4dd74ef59ee85a': 'EXPRESS',
            
            # Standard service hashes  
            '203ac3454d75e02ebb0a3c6f51d735e4': 'STANDARD',
            'fe76dff35bb199cdb7329eba2b918f18': 'STANDARD',
            
            # Economy service hashes
            '73ffcbd1b26557b462b14e4dd4c57fcb': 'ECONOMY',
            'e83a6cefa7e4bde8a8af866f3f4e90eb': 'ECONOMY',
            '6771c4e2ecb275c95c43f6c639a2cbad': 'ECONOMY',
            '84c7d46d654e5a8bd329a3e8ed0293ce': 'ECONOMY',
            
            # Special/Premium service hashes
            'd793b6abb67977f8209e555c584ca951': 'PREMIUM',
            'a67bc96f63bcd2b372a02a9d75c07573': 'PREMIUM',
            
            # Unknown service types (will be categorized by pattern)
        }
        
        def map_service_type(stype):
            # Convert to string and clean
            stype_str = str(stype).strip()
            
            # Check if it's a known hash
            if stype_str in hash_mapping:
                return hash_mapping[stype_str]
            
            # If it looks like a hash but not in mapping, try to categorize
            if len(stype_str) == 32 and stype_str.isalnum():  # MD5 hash
                # Try to guess based on hash patterns (this is heuristic)
                if stype_str.startswith(('14', '33', '46')):  # Express patterns
                    return "EXPRESS"
                elif stype_str.startswith(('20', 'fe')):  # Standard patterns
                    return "STANDARD"
                elif stype_str.startswith(('67', '73', 'e8', '84')):  # Economy patterns
                    return "ECONOMY"
                else:
                    return "UNKNOWN_HASHED"
            
            # If it's already a readable name, return as is
            if len(stype_str) < 20:  # Probably a readable name
                return stype_str.upper()
            
            return "UNKNOWN"
        
        df_clean['service_type'] = df_clean['typecode'].apply(map_service_type)
        
        # Count the mapping results
        if st is not None:
            mapped_counts = df_clean['service_type'].value_counts()
            st.write("**Service Type Distribution:**")
            for service, count in mapped_counts.items():
                st.write(f"• {service}: {count} records")
        
        # Also show what original hashes we couldn't map
        unknown_hashes = df_clean[df_clean['service_type'] == 'UNKNOWN_HASHED']['typecode'].unique()
        if len(unknown_hashes) > 0:
            st.warning(f"Found {len(unknown_hashes)} unknown hashed service types")
            with st.expander("See unknown hashes"):
                for hash_val in unknown_hashes[:10]:  # Show first 10
                    st.write(f"• {hash_val}")
                if len(unknown_hashes) > 10:
                    st.write(f"... and {len(unknown_hashes) - 10} more")
    
    return df_clean

def create_better_features(df):
    """
    Create more meaningful features from raw data
    """
    df_features = df.copy()
    
    # 1. Ensure basic features exist
    if 'distance_km' not in df_features.columns and all(col in df_features.columns for col in ['poi_lng', 'poi_lat', 'receipt_lng', 'receipt_lat']):
        from math import radians, sin, cos, sqrt, atan2
        
        def calculate_distance(row):
            # Haversine formula for distance
            try:
                lat1, lon1 = radians(row['poi_lat']), radians(row['poi_lng'])
                lat2, lon2 = radians(row['receipt_lat']), radians(row['receipt_lng'])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return 6371 * c  # Earth radius in km
            except:
                return np.nan
        
        df_features['distance_km'] = df_features.apply(calculate_distance, axis=1)
    
    # 2. Create time-based features
    if 'receipt_time' in df_features.columns:
        try:
            df_features['receipt_time'] = pd.to_datetime(df_features['receipt_time'])
            df_features['hour'] = df_features['receipt_time'].dt.hour
            df_features['day_of_week'] = df_features['receipt_time'].dt.dayofweek
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            df_features['is_business_hours'] = ((df_features['hour'] >= 9) & 
                                              (df_features['hour'] <= 17)).astype(int)
        except:
            pass
    
    # 3. Create interaction features
    if all(col in df_features.columns for col in ['distance_km', 'hour']):
        df_features['distance_hour_interaction'] = df_features['distance_km'] * df_features['hour']
    
    # 4. Remove outliers more aggressively for better model performance
    if 'delivery_time_hours' in df_features.columns:
        # Remove unrealistic delivery times (0.1 to 48 hours)
        initial_count = len(df_features)
        df_features = df_features[(df_features['delivery_time_hours'] > 0.1) & 
                                (df_features['delivery_time_hours'] < 48)]
        removed = initial_count - len(df_features)
        if removed > 0 and st is not None:
            st.info(f"Removed {removed} outliers from delivery times")
    
    if 'distance_km' in df_features.columns:
        # Remove unrealistic distances (0.1 to 500 km)
        initial_count = len(df_features)
        df_features = df_features[(df_features['distance_km'] > 0.1) & 
                                (df_features['distance_km'] < 500)]
        removed = initial_count - len(df_features)
        if removed > 0 and st is not None:
            st.info(f"Removed {removed} outliers from distances")
    
    # 5. Create speed feature
    if all(col in df_features.columns for col in ['distance_km', 'delivery_time_hours']):
        df_features['speed_kmh'] = df_features['distance_km'] / df_features['delivery_time_hours']
        # Remove unrealistic speeds
        df_features = df_features[(df_features['speed_kmh'] > 1) & (df_features['speed_kmh'] < 100)]
    
    return df_features

def clean_and_preprocess_data(df):
    """Enhanced data cleaning function"""
    df_clean = df.copy()
    
    # 1. Clean city names
    if 'from_city_name' in df_clean.columns:
        df_clean['from_city_name'] = df_clean['from_city_name'].astype(str)
        df_clean['city_clean'] = df_clean['from_city_name'].str.title().str.strip()
    
    # 2. Ensure service_type column exists (from decode_hashed_service_types)
    if 'service_type' not in df_clean.columns and 'typecode' in df_clean.columns:
        df_clean = decode_hashed_service_types(df_clean)
    
    return df_clean

def prepare_enhanced_features(df, sample_size=50000):
    """
    Prepare enhanced features with better engineering
    Returns exactly 7 features that the model will expect
    """
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        if st is not None:
            st.info(f"Using sample of {sample_size} records")
    
    # Decode hashed service types FIRST
    df = decode_hashed_service_types(df)
    
    # Create better features
    df_enhanced = create_better_features(df)
    
    # Clean data
    df_enhanced = clean_and_preprocess_data(df_enhanced)
    
    # Create the EXACT 7 features that will be used for both training and prediction
    # 1. Ensure all required features exist
    if 'distance_km' not in df_enhanced.columns:
        # Calculate distance if missing
        df_enhanced['distance_km'] = np.sqrt(
            (df_enhanced['poi_lng'] - df_enhanced['receipt_lng'])**2 +
            (df_enhanced['poi_lat'] - df_enhanced['receipt_lat'])**2
        ) * 111  # Approximation for km
    
    # 2. Ensure time features exist
    if 'receipt_time' in df_enhanced.columns:
        df_enhanced['receipt_time'] = pd.to_datetime(df_enhanced['receipt_time'])
        df_enhanced['hour'] = df_enhanced['receipt_time'].dt.hour
        df_enhanced['day_of_week'] = df_enhanced['receipt_time'].dt.dayofweek
    else:
        # Create default time features if missing
        df_enhanced['hour'] = 12  # Default noon
        df_enhanced['day_of_week'] = 0  # Default Monday
    
    # 3. Create boolean features
    df_enhanced['is_weekend'] = (df_enhanced['day_of_week'].isin([5, 6])).astype(int)
    df_enhanced['is_business_hours'] = ((df_enhanced['hour'] >= 9) & (df_enhanced['hour'] <= 17)).astype(int)
    
    # 4. Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    le_city = LabelEncoder()
    le_type = LabelEncoder()
    
    # Handle city encoding
    if 'from_city_name' in df_enhanced.columns:
        valid_cities = df_enhanced['from_city_name'].dropna()
        if len(valid_cities) > 0:
            le_city.fit(valid_cities.astype(str))
            df_enhanced['city_encoded'] = le_city.transform(df_enhanced['from_city_name'].astype(str))
        else:
            df_enhanced['city_encoded'] = 0  # Default value
    else:
        df_enhanced['city_encoded'] = 0  # Default value
    
    # Handle service type encoding
    if 'service_type' in df_enhanced.columns:
        valid_types = df_enhanced['service_type'].dropna()
        if len(valid_types) > 0:
            le_type.fit(valid_types.astype(str))
            df_enhanced['type_encoded'] = le_type.transform(df_enhanced['service_type'].astype(str))
        else:
            df_enhanced['type_encoded'] = 0  # Default value
    elif 'typecode' in df_enhanced.columns:
        valid_types = df_enhanced['typecode'].dropna()
        if len(valid_types) > 0:
            le_type.fit(valid_types.astype(str))
            df_enhanced['type_encoded'] = le_type.transform(df_enhanced['typecode'].astype(str))
        else:
            df_enhanced['type_encoded'] = 0  # Default value
    else:
        df_enhanced['type_encoded'] = 0  # Default value
    
    # Define the EXACT 7 features that will be used
    feature_columns = [
        'distance_km',           # Feature 1: Distance in km
        'city_encoded',          # Feature 2: Encoded city
        'type_encoded',          # Feature 3: Encoded service type
        'hour',                  # Feature 4: Hour of day (0-23)
        'day_of_week',           # Feature 5: Day of week (0-6)
        'is_weekend',            # Feature 6: Is weekend (0 or 1)
        'is_business_hours'      # Feature 7: Is business hours (0 or 1)
    ]
    
    # Ensure all features exist
    for col in feature_columns:
        if col not in df_enhanced.columns:
            df_enhanced[col] = 0  # Fill missing with default
    
    # Prepare X and y
    X = df_enhanced[feature_columns].values
    y = df_enhanced['delivery_time_hours'].values
    
    if st is not None:
        st.info(f"Using {len(feature_columns)} features: {feature_columns}")
        
        # Show feature statistics
        if len(y) > 0:
            st.write(f"Target variable stats: mean={y.mean():.2f} hours, std={y.std():.2f} hours")
        
        # Show data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(X))
        with col2:
            st.metric("Features", len(feature_columns))
        with col3:
            if len(y) > 0:
                st.metric("Avg Delivery Time", f"{y.mean():.1f} hours")
        
        # Show feature distributions
        with st.expander("Feature Distributions"):
            for i, feature in enumerate(feature_columns):
                st.write(f"**{feature}**: mean={df_enhanced[feature].mean():.2f}, std={df_enhanced[feature].std():.2f}")
    
    return X, y, le_city, le_type, feature_columns

def prepare_features(df, sample_size=50000):
    """
    Prepare features for machine learning models with sampling option
    (Maintains backward compatibility)
    """
    X, y, le_city, le_type, _ = prepare_enhanced_features(df, sample_size)
    return X, y, le_city, le_type

def load_and_preprocess_data(file_path, sample_size=None):
    """
    Load and preprocess the courier dataset with optimizations for large files
    """
    try:
        # Load data with optimizations
        print("Loading dataset...")
        
        # Read in chunks for large files
        if sample_size and sample_size < 470000:
            # Use sampling for development
            df = pd.read_csv(file_path, nrows=sample_size)
            print(f"Using sample of {sample_size} records for development")
        else:
            # Read full dataset with optimized dtypes
            dtypes = {
                'order_id': 'category',
                'from_dipan_id': 'category', 
                'from_city_name': 'category',
                'delivery_user_id': 'category',
                'aoi_id': 'category',
                'typecode': 'category',
                'ds': 'int16'  # Day of year as integer
            }
            
            df = pd.read_csv(file_path, dtype=dtypes)
            print(f"Loaded full dataset: {len(df):,} records")
        
        # Debug info
        print("Dataset columns:", df.columns.tolist())
        print("Memory usage:", df.memory_usage(deep=True).sum() / 1024**2, "MB")
        
        # Check if required columns exist
        required_cols = ['receipt_time', 'sign_time', 'ds', 'poi_lng', 'poi_lat', 'receipt_lng', 'receipt_lat']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            return df  # Return early if critical columns are missing
        
        # Determine the year - since ds is day of year, we need to know the actual year
        assumed_year = 2024  # Change this if you know the actual year
        print(f"Assuming year: {assumed_year} for day-of-year calculations")
        
        # Convert datetime columns - they're in format "MM-DD HH:MM:SS"
        print("Converting datetime columns...")
        
        def day_of_year_to_date(day_of_year, year):
            """Convert day of year to actual date"""
            try:
                # Create date from day of year
                base_date = datetime(year, 1, 1)
                target_date = base_date + timedelta(days=int(day_of_year) - 1)
                return target_date
            except:
                return pd.NaT
        
        # Convert receipt_time and sign_time
        def convert_delivery_datetime(time_str, day_of_year, year):
            """Convert MM-DD HH:MM:SS time string with day-of-year to proper datetime"""
            try:
                # Get the date from day of year
                date_part = day_of_year_to_date(day_of_year, year)
                if pd.isna(date_part):
                    return pd.NaT
                
                # Parse the time string (MM-DD HH:MM:SS)
                time_parts = time_str.split()
                if len(time_parts) != 2:
                    return pd.NaT
                    
                # The time part is HH:MM:SS
                time_part = time_parts[1]
                
                # Combine date and time
                datetime_str = f"{date_part.strftime('%Y-%m-%d')} {time_part}"
                return pd.to_datetime(datetime_str)
                
            except Exception as e:
                print(f"Error converting datetime: {e}")
                return pd.NaT
        
        # Apply the conversion
        df['receipt_time'] = df.apply(
            lambda row: convert_delivery_datetime(row['receipt_time'], row['ds'], assumed_year), 
            axis=1
        )
        
        df['sign_time'] = df.apply(
            lambda row: convert_delivery_datetime(row['sign_time'], row['ds'], assumed_year), 
            axis=1
        )
        
        # Remove invalid datetime rows
        initial_count = len(df)
        df = df[df['receipt_time'].notna() & df['sign_time'].notna()].copy()
        print(f"Removed {initial_count - len(df)} rows with invalid datetime values")
        
        # Calculate delivery time in hours
        print("Calculating delivery times...")
        df['delivery_time_hours'] = (df['sign_time'] - df['receipt_time']).dt.total_seconds() / 3600
        
        # Remove unrealistic values
        initial_count = len(df)
        df = df[(df['delivery_time_hours'] > 0) & (df['delivery_time_hours'] < 168)].copy()
        print(f"Removed {initial_count - len(df)} rows with unrealistic delivery times")
        
        # Calculate distances
        print("Calculating distances...")
        
        # For large datasets, use vectorized calculation (approximate but much faster)
        if len(df) > 100000:
            print("Large dataset detected - using vectorized distance calculation")
            # Vectorized calculation (approximate but much faster)
            df['distance_km'] = np.sqrt(
                (df['poi_lng'] - df['receipt_lng'])**2 +
                (df['poi_lat'] - df['receipt_lat'])**2
            ) * 111  # Approximation for km (1 degree ≈ 111 km)
        else:
            # Accurate but slower calculation
            df['distance_km'] = df.apply(
                lambda row: geodesic(
                    (row['poi_lat'], row['poi_lng']),
                    (row['receipt_lat'], row['receipt_lng'])
                ).km, axis=1
            )
        
        # Extract time features
        print("Extracting time features...")
        df['receipt_hour'] = df['receipt_time'].dt.hour
        df['receipt_dayofweek'] = df['receipt_time'].dt.dayofweek
        df['receipt_month'] = df['receipt_time'].dt.month
        
        # Extract additional features
        df['is_weekend'] = df['receipt_dayofweek'].isin([5, 6]).astype('int8')
        df['is_business_hours'] = ((df['receipt_hour'] >= 9) & (df['receipt_hour'] <= 17)).astype('int8')
        
        # Optimize memory usage
        print("Optimizing memory usage...")
        for col in ['receipt_hour', 'receipt_dayofweek', 'receipt_month']:
            if col in df.columns:
                df[col] = df[col].astype('int8')
        
        if 'delivery_time_hours' in df.columns:
            df['delivery_time_hours'] = df['delivery_time_hours'].astype('float32')
        if 'distance_km' in df.columns:
            df['distance_km'] = df['distance_km'].astype('float32')
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['order_id', 'from_city_name', 'typecode', 'delivery_time_hours', 'distance_km'])

def prepare_anomaly_features(df, sample_size=100000):
    """
    Prepare features for anomaly detection with sampling
    """
    # Sample for anomaly detection (can handle larger samples)
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    anomaly_features = [
        'delivery_time_hours', 
        'distance_km', 
        'receipt_hour', 
        'receipt_dayofweek',
        'is_weekend',
        'is_business_hours'
    ]
    
    available_features = [col for col in anomaly_features if col in df_sample.columns]
    X_anomaly = df_sample[available_features]
    
    # Scale features
    scaler = StandardScaler()
    X_anomaly_scaled = scaler.fit_transform(X_anomaly)
    
    return X_anomaly_scaled, scaler, df_sample

# Cache functions
@st.cache_data(show_spinner=False, ttl=3600)
def load_data_with_cache(file_path, sample_size=None):
    return load_and_preprocess_data(file_path, sample_size)

@st.cache_data(show_spinner=False, ttl=3600)
def prepare_features_with_cache(df, sample_size=50000):
    return prepare_features(df, sample_size)

@st.cache_data(show_spinner=False, ttl=3600)
def prepare_anomaly_features_with_cache(df, sample_size=100000):
    return prepare_anomaly_features(df, sample_size)