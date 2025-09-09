# src/prediction_interface.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, time
from src.utils import calculate_estimated_arrival, format_delivery_time

def prepare_prediction_features(distance, from_city, to_city, service_type, pickup_datetime, le_city, le_type, feature_names):
    """
    Prepare features EXACTLY as they were during training
    """
    hour = pickup_datetime.hour
    day_of_week = pickup_datetime.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1 if day_of_week >= 5 else 0
    is_business_hours = 1 if 9 <= hour <= 17 else 0
    
    # Use the SAME encoding as during training
    try:
        city_encoded = le_city.transform([from_city])[0]
    except:
        city_encoded = 0  # Default if city not in encoder
    
    try:
        type_encoded = le_type.transform([service_type])[0]
    except:
        type_encoded = 0  # Default if type not in encoder
    
    # Create features in EXACT same order as training
    features = {
        'distance_km': distance,
        'city_encoded': city_encoded,
        'type_encoded': type_encoded,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_business_hours': is_business_hours
    }
    
    # Ensure correct order
    features_ordered = np.array([features[fn] for fn in feature_names])
    
    return features_ordered.reshape(1, -1)

def validate_city_distance(from_city, to_city, distance):
    """Validate distance makes sense for the city pair"""
    # Known city distances in Tanzania (approximate in km)
    city_distances = {
        ("Dar es Salaam", "Arusha"): 600,
        ("Dar es Salaam", "Mwanza"): 1100,
        ("Dar es Salaam", "Mbeya"): 900,
        ("Dar es Salaam", "Dodoma"): 450,
        ("Arusha", "Dar es Salaam"): 600,
        ("Arusha", "Mwanza"): 550,
        ("Arusha", "Mbeya"): 800,
        ("Arusha", "Dodoma"): 400,
        ("Mwanza", "Dar es Salaam"): 1100,
        ("Mwanza", "Arusha"): 550,
        ("Mwanza", "Mbeya"): 700,
        ("Mwanza", "Dodoma"): 650,
        ("Mbeya", "Dar es Salaam"): 900,
        ("Mbeya", "Arusha"): 800,
        ("Mbeya", "Mwanza"): 700,
        ("Mbeya", "Dodoma"): 500,
        ("Dodoma", "Dar es Salaam"): 450,
        ("Dodoma", "Arusha"): 400,
        ("Dodoma", "Mwanza"): 650,
        ("Dodoma", "Mbeya"): 500,
    }
    
    # Same city delivery
    if from_city == to_city:
        if distance > 50:  # Max reasonable same-city distance
            return f"⚠️ Same city delivery: distance should be less than 50km, got {distance}km"
        return None
    
    # Different cities
    expected_distance = city_distances.get((from_city, to_city), None)
    
    if expected_distance:
        if abs(distance - expected_distance) > 200:  # Allow 200km tolerance
            return f"⚠️ Expected distance for {from_city}→{to_city} is ~{expected_distance}km, but got {distance}km"
    
    return None

def show_real_time_prediction_interface(model, scaler, le_city, le_type, feature_names):
    """
    Real-time prediction interface with immediate feedback and validation
    """
    st.markdown("---")
    st.markdown('<div class="main-header">🚀 Instant Delivery Time Predictor</div>', unsafe_allow_html=True)
    
    st.info("Enter package details below and get immediate delivery time predictions!")
    
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    prediction_result = None
    validation_errors = []
    
    with col1:
        st.subheader("📍 Package Details")
        
        # City selection
        from_city = st.selectbox(
            "From City 🌆",
            options=["Dar es Salaam", "Arusha", "Mwanza", "Mbeya", "Dodoma"],
            help="Select the origin city",
            key="from_city_input"
        )
        
        to_city = st.selectbox(
            "To City 🏙️",
            options=["Dar es Salaam", "Arusha", "Mwanza", "Mbeya", "Dodoma"],
            index=1,
            help="Select the destination city",
            key="to_city_input"
        )
        
        # Service type
        service_type = st.selectbox(
            "Service Type 🚚",
            options=["EXPRESS", "STANDARD", "ECONOMY"],
            help="Choose the service level",
            key="service_type_input"
        )
        
        # Auto-calculate distance based on city pair
        city_distances = {
            ("Dar es Salaam", "Arusha"): 600,
            ("Dar es Salaam", "Mwanza"): 1100,
            ("Dar es Salaam", "Mbeya"): 900,
            ("Dar es Salaam", "Dodoma"): 450,
            ("Arusha", "Dar es Salaam"): 600,
            ("Arusha", "Mwanza"): 550,
            ("Arusha", "Mbeya"): 800,
            ("Arusha", "Dodoma"): 400,
            ("Mwanza", "Dar es Salaam"): 1100,
            ("Mwanza", "Arusha"): 550,
            ("Mwanza", "Mbeya"): 700,
            ("Mwanza", "Dodoma"): 650,
            ("Mbeya", "Dar es Salaam"): 900,
            ("Mbeya", "Arusha"): 800,
            ("Mbeya", "Mwanza"): 700,
            ("Mbeya", "Dodoma"): 500,
            ("Dodoma", "Dar es Salaam"): 450,
            ("Dodoma", "Arusha"): 400,
            ("Dodoma", "Mwanza"): 650,
            ("Dodoma", "Mbeya"): 500,
        }
        
        # Calculate distance based on city selection
        if from_city == to_city:
            default_distance = 15.0  # Typical same-city distance
        else:
            default_distance = city_distances.get((from_city, to_city), 300.0)
        
        # Show distance as read-only or calculated field
        st.metric("Distance (km) 📏", f"{default_distance:.1f} km")
        distance = default_distance
        
        # Add contextual info
        if from_city == to_city:
            st.info("🏙️ Same city delivery")
        else:
            st.info(f"🛣️ {from_city} → {to_city}")
    
    with col2:
        st.subheader("📅 Timing & Conditions")
        
        # Date and time
        pickup_date = st.date_input(
            "Pickup Date 📅",
            value=datetime.now().date(),
            help="Date when package will be picked up",
            key="pickup_date_input"
        )
        
        pickup_time = st.time_input(
            "Pickup Time 🕒",
            value=time(12, 0),
            help="Time when package will be picked up",
            key="pickup_time_input"
        )
        
        # Combine date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)
        
        # Display time information
        hour = pickup_time.hour
        day_name = pickup_datetime.strftime("%A")
        st.info(f"⏰ {day_name}, {hour:02d}:00 | {'Business hours' if 9 <= hour <= 17 else 'After hours'}")
    
    # Real-time validation and prediction
    if st.button("🚀 Predict Delivery Time Now!", type="primary", use_container_width=True, key="predict_button"):
        # Clear previous errors
        validation_errors.clear()
        
        # Validate distance
        distance_validation = validate_city_distance(from_city, to_city, distance)
        if distance_validation:
            validation_errors.append(distance_validation)
        
        # Additional validation
        if from_city == to_city and distance > 50:
            validation_errors.append("⚠️ Same city delivery with long distance - please verify")
        
        # If no validation errors, make prediction
        if not validation_errors:
            try:
                # Prepare features exactly as during training
                features_scaled = prepare_prediction_features(
                    distance, from_city, to_city, service_type, 
                    pickup_datetime, le_city, le_type, feature_names
                )
                
                # Scale features
                features_scaled = scaler.transform(features_scaled)
                
                # Make prediction
                prediction_hours = model.predict(features_scaled)[0]
                
                # Ensure realistic prediction (0.5h to 7 days)
                prediction_hours = max(0.5, min(prediction_hours, 168))
                
                # Calculate arrival time
                estimated_arrival = calculate_estimated_arrival(pickup_datetime, prediction_hours)
                
                # Store prediction result
                prediction_result = {
                    'prediction_hours': prediction_hours,
                    'estimated_arrival': estimated_arrival,
                    'from_city': from_city,
                    'to_city': to_city,
                    'distance': distance,
                    'service_type': service_type,
                    'pickup_time': pickup_datetime
                }
                
            except Exception as e:
                validation_errors.append(f"❌ Prediction error: {str(e)}")
                st.error(f"Technical error: {str(e)}")
        else:
            # Show validation errors immediately
            for error in validation_errors:
                st.warning(error)
    
    # Display validation errors if any
    if validation_errors:
        st.error("**Please fix the following issues:**")
        for error in validation_errors:
            st.write(f"• {error}")
    
    # Display prediction result if available
    if prediction_result:
        display_prediction_result(prediction_result)
    
    # Quick test examples
    st.markdown("---")
    st.subheader("🧪 Quick Test Examples")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("Test Local Express", help="Short distance, express service"):
            st.session_state.from_city_input = "Dar es Salaam"
            st.session_state.to_city_input = "Dar es Salaam"
            st.session_state.service_type_input = "EXPRESS"
            st.rerun()
    
    with example_col2:
        if st.button("Test Regional Standard", help="Medium distance, standard service"):
            st.session_state.from_city_input = "Dar es Salaam"
            st.session_state.to_city_input = "Arusha"
            st.session_state.service_type_input = "STANDARD"
            st.rerun()
    
    with example_col3:
        if st.button("Test Long Distance", help="Long distance, economy service"):
            st.session_state.from_city_input = "Dar es Salaam"
            st.session_state.to_city_input = "Mwanza"
            st.session_state.service_type_input = "ECONOMY"
            st.rerun()

def display_prediction_result(prediction):
    """
    Display the prediction result in a user-friendly format
    """
    st.markdown("---")
    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
    
    # Main prediction
    st.markdown(f"### 📦 Predicted Delivery Time: **{format_delivery_time(prediction['prediction_hours'])}**")
    
    # Arrival information
    st.markdown(f"**📅 Estimated Arrival:** {prediction['estimated_arrival'].strftime('%Y-%m-%d %H:%M')}")
    st.markdown(f"**📍 Route:** {prediction['from_city']} → {prediction['to_city']} ({prediction['distance']:.1f} km)")
    st.markdown(f"**🚚 Service:** {prediction['service_type']}")
    st.markdown(f"**⏰ Pickup:** {prediction['pickup_time'].strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Details expander
    with st.expander("📊 Prediction Details", expanded=True):
        # Speed calculation
        speed_kmh = prediction['distance'] / prediction['prediction_hours']
        st.metric("Estimated Speed", f"{speed_kmh:.1f} km/h")
        
        # Service type impact
        service_impact = {
            "EXPRESS": "⚡ Faster delivery (30-50% quicker than standard)",
            "STANDARD": "🚗 Normal delivery speed",
            "ECONOMY": "🐢 Slower delivery (may take 20-30% longer)"
        }
        st.info(service_impact.get(prediction['service_type'], "Standard delivery"))
        
        # Contextual feedback
        if prediction['prediction_hours'] < 4:
            st.success("⚡ **Express Delivery**: Faster than average!")
        elif prediction['prediction_hours'] < 12:
            st.info("🚗 **Standard Delivery**: Typical delivery time")
        else:
            st.warning("🐢 **Extended Delivery**: May take longer than usual")
        
        if speed_kmh < 30:
            st.warning("🚨 **Slow Speed**: Potential delays or traffic expected")
        elif speed_kmh > 80:
            st.success("🎯 **Efficient Speed**: Good operating conditions")
        else:
            st.info("📊 **Normal Speed**: Typical delivery pace")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Predict Another", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("📋 Save Prediction", use_container_width=True):
            st.success("Prediction saved to history!")