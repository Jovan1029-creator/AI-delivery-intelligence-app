# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from src.data_processing import prepare_enhanced_features, decode_hashed_service_types
from src.model_training import train_enhanced_delivery_time_model
warnings.filterwarnings('ignore')


# Set page configuration
st.set_page_config(
    page_title="AI-Powered Delivery Intelligence System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
            
    .stForm {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e9ecef;
}

.stButton>button {
    width: 100%;
    border-radius: 5px;
}

.stTextInput>div>div>input {
    border-radius: 5px;
}

.stSelectbox>div>div>select {
    border-radius: 5px;
}
            
    .anomaly-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
        margin-bottom: 1rem;
    }
    .map-container {
        border-radius: 0.5rem;
        overflow: hidden;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Cached data loading functions
@st.cache_data(show_spinner=False, ttl=3600)
def load_data_with_cache(file_path, sample_size=None):
    from src.data_processing import load_and_preprocess_data
    return load_and_preprocess_data(file_path, sample_size)

@st.cache_data(show_spinner=False, ttl=3600)
def prepare_features_with_cache(df, sample_size=50000):
    from src.data_processing import prepare_features
    return prepare_features(df, sample_size)

@st.cache_data(show_spinner=False, ttl=3600)  
def prepare_anomaly_features_with_cache(df, sample_size=100000):
    from src.data_processing import prepare_anomaly_features
    return prepare_anomaly_features(df, sample_size)

@st.cache_resource(show_spinner=False, ttl=3600)
def train_models_with_cache(X, y, X_anomaly_scaled, df_anomaly):
    from src.model_training import train_delivery_time_model
    from src.anomaly_detection import detect_anomalies, analyze_anomalies
    
    model, X_test, y_test, y_pred, mae, rmse, r2 = train_delivery_time_model(X, y)
    anomalies = detect_anomalies(X_anomaly_scaled)
    df_with_anomalies, anomaly_by_city, anomaly_by_type = analyze_anomalies(df_anomaly, anomalies)
    
    return model, X_test, y_test, y_pred, mae, rmse, r2, anomalies, df_with_anomalies, anomaly_by_city, anomaly_by_type

def show_dashboard(df, anomalies):
    st.markdown('<div class="main-header">Delivery Intelligence Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Deliveries", f"{len(df):,}")
    
    with col2:
        avg_time = df['delivery_time_hours'].mean()
        st.metric("Avg Delivery Time", f"{avg_time:.2f} hours")
    
    with col3:
        anomaly_count = (anomalies == -1).sum() if hasattr(anomalies, 'sum') else 0
        st.metric("Anomalies Detected", f"{anomaly_count}")
    
    with col4:
        on_time_rate = np.mean(df['delivery_time_hours'] <= 24)
        st.metric("On-Time Delivery Rate", f"{on_time_rate*100:.1f}%")
    
    # Map Visualization
    st.markdown('<div class="sub-header">📍 Delivery Operations Map</div>', unsafe_allow_html=True)
    show_basic_map(df)
    
    # Visualizations
    from src.visualization import create_delivery_time_histogram, create_city_performance_chart
    st.plotly_chart(create_delivery_time_histogram(df), use_container_width=True)
    st.plotly_chart(create_city_performance_chart(df), use_container_width=True)

def show_prediction_section():
    """
    Delivery Time Prediction section with user input form and model predictions
    """
    st.markdown('<div class="main-header">Delivery Time Prediction</div>', unsafe_allow_html=True)
    
    # Check if model and required components are available
    if (st.session_state.model is None or 
        st.session_state.le_city is None or 
        st.session_state.le_type is None):
        st.error("❌ Model not ready yet. Please wait for training to complete or check your data.")
        return
    
    # Warning about poor model performance
    if hasattr(st.session_state, 'r2') and st.session_state.r2 < 0.3:
        st.warning(f"""
        ⚠️ **Model Performance Alert**
        
        Your model has low predictive power (R² = {st.session_state.r2:.3f}). This could be due to:
        
        - Poor quality or inconsistent data
        - Incorrect feature engineering
        - Hashed or encoded service types/cities instead of readable names
        - Insufficient training data
        
        Predictions may not be accurate or responsive to input changes.
        """)
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Model Performance Metrics
        st.markdown("### 📊 Model Performance")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        if (hasattr(st.session_state, 'mae') and 
            hasattr(st.session_state, 'rmse') and 
            hasattr(st.session_state, 'r2')):
            st.metric("Mean Absolute Error", f"{st.session_state.mae:.2f} hours")
            st.metric("Root Mean Squared Error", f"{st.session_state.rmse:.2f} hours")
            st.metric("R² Score", f"{st.session_state.r2:.3f}")
            
            # Interpretation of R² score
            if st.session_state.r2 > 0.7:
                st.success("✅ Excellent predictive power")
            elif st.session_state.r2 > 0.5:
                st.info("📈 Good predictive power")
            elif st.session_state.r2 > 0.3:
                st.warning("📉 Moderate predictive power")
            else:
                st.error("❌ Poor predictive power - check your data!")
        else:
            st.info("⏳ Performance metrics being calculated...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality insights
        st.markdown("### 🔍 Data Insights")
        if hasattr(st.session_state, 'df'):
            df = st.session_state.df
            st.write(f"• Total records: {len(df):,}")
            
            if 'from_city_name' in df.columns:
                unique_cities = df['from_city_name'].nunique()
                st.write(f"• Unique cities: {unique_cities}")
                if unique_cities <= 10:
                    cities_sample = df['from_city_name'].unique()[:5]
                    st.write(f"  Sample: {', '.join(map(str, cities_sample))}")
            
            if 'service_type' in df.columns:
                unique_services = df['service_type'].nunique()
                st.write(f"• Unique service types: {unique_services}")
                if unique_services <= 10:
                    services_sample = df['service_type'].unique()[:5]
                    st.write(f"  Sample: {', '.join(map(str, services_sample))}")
            elif 'typecode' in df.columns:
                unique_services = df['typecode'].nunique()
                st.write(f"• Unique service types: {unique_services}")
            
            if 'delivery_time_hours' in df.columns:
                st.write(f"• Avg delivery time: {df['delivery_time_hours'].mean():.2f} hours")
                st.write(f"• Delivery time range: {df['delivery_time_hours'].min():.1f} to {df['delivery_time_hours'].max():.1f} hours")
        
        # Debug data quality button
        if st.button("🛠️ Debug Data Quality", key="debug_data_btn"):
            debug_data_quality()
        
        # Actual vs Predicted Plot (if test data is available)
        if (hasattr(st.session_state, 'X_test') and 
            hasattr(st.session_state, 'y_test') and 
            st.session_state.X_test is not None):
            try:
                from src.visualization import create_actual_vs_predicted_plot
                y_pred = st.session_state.model.predict(st.session_state.X_test)
                fig = create_actual_vs_predicted_plot(st.session_state.y_test, y_pred)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate performance chart: {str(e)}")
        else:
            st.info("📈 Performance chart will appear after model evaluation")
    
    with col2:
        st.markdown("### 📋 Predict New Delivery")
        st.info("Enter details below to predict delivery time for a new parcel")
        
        # Get available service types from the trained encoder
        available_service_types = list(st.session_state.le_type.classes_)
        available_service_types = [str(x) for x in available_service_types if pd.notna(x)]
        
        # Get available cities from the trained encoder
        available_cities = list(st.session_state.le_city.classes_)
        available_cities = [str(x) for x in available_cities if pd.notna(x)]
        
        # Store prediction result in session state
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
        
        # User input form
        with st.form("prediction_form"):
            # City selection - only show cities the model knows about
            city = st.selectbox(
                "Destination City 🌆",
                options=available_cities,
                help="Select the destination city for delivery"
            )
            
            # Service type - only show service types the model knows about
            service_type = st.selectbox(
                "Service Type 🚚",
                options=available_service_types,
                help="Choose the service level"
            )
            
            # Distance with visual feedback
            distance = st.slider(
                "Distance (km) 📏",
                min_value=0.1,
                max_value=200.0,
                value=15.0,
                step=0.5,
                help="Approximate distance between pickup and delivery locations"
            )
            
            # Show distance context
            if distance < 5:
                st.caption("🏙️ Short distance (within city)")
            elif distance < 20:
                st.caption("🚗 Medium distance (city to suburb)")
            else:
                st.caption("🛣️ Long distance (between cities)")
            
            # Time and date inputs
            st.markdown("**🕒 Time & Date Information**")
            time_col1, time_col2 = st.columns(2)
            
            with time_col1:
                hour = st.slider(
                    "Pickup Hour", 
                    0, 23, 12,
                    help="Hour of day when package is received (0-23)"
                )
            
            with time_col2:
                day_of_week = st.selectbox(
                    "Day of Week",
                    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=0  # Default to Monday
                )
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Predict Delivery Time",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                try:
                    # Input validation
                    from src.utils import validate_prediction_inputs, apply_prediction_sanity_checks
                    errors, warnings = validate_prediction_inputs(distance, city, service_type, hour, day_of_week)
                    
                    if errors:
                        for error in errors:
                            st.error(error)
                        return
                    
                    for warning in warnings:
                        st.warning(warning)
                    
                    # Convert user input to model features
                    city_encoded = st.session_state.le_city.transform([city])[0]
                    type_encoded = st.session_state.le_type.transform([service_type])[0]
                    day_encoded = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
                    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
                    is_business_hours = 1 if 9 <= hour <= 17 else 0
                    
                    # Create feature array with EXACTLY 7 features (matching training)
                    features = np.array([[
                        distance,           # distance_km (Feature 1)
                        city_encoded,       # city_encoded (Feature 2)
                        type_encoded,       # type_encoded (Feature 3)
                        hour,               # hour (Feature 4)
                        day_encoded,        # day_of_week (Feature 5)
                        is_weekend,         # is_weekend (Feature 6)
                        is_business_hours,  # is_business_hours (Feature 7)
                    ]])
                    
                    # Validate feature count
                    if len(features[0]) != st.session_state.model.n_features_in_:
                        st.error(f"❌ Feature mismatch! Model expects {st.session_state.model.n_features_in_} features, but got {len(features[0])}")
                        st.write("Features provided:", features[0])
                        st.info("Please check your model training code to see what features were used.")
                    else:
                        # Make prediction
                        raw_prediction = st.session_state.model.predict(features)[0]
                        
                        # Apply sanity checks
                        final_prediction, sanity_warnings = apply_prediction_sanity_checks(
                            raw_prediction, distance, service_type, hour, day_of_week
                        )
                        
                        # Show warnings if any
                        for warning in sanity_warnings:
                            st.warning(warning)
                        
                        # Show realistic time bounds
                        from src.utils import calculate_realistic_time_bounds
                        min_time, max_time, expected_time = calculate_realistic_time_bounds(
                            distance, service_type, hour, day_of_week
                        )
                        
                        with st.expander("📊 Realistic Time Expectations"):
                            st.write(f"**Based on distance and service type:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Minimum Expected", f"{min_time:.1f}h")
                            with col2:
                                st.metric("Typical Expected", f"{expected_time:.1f}h")
                            with col3:
                                st.metric("Maximum Expected", f"{max_time:.1f}h")
                        
                        # Store prediction in session state
                        st.session_state.prediction_result = {
                            'prediction': final_prediction,
                            'raw_prediction': raw_prediction,
                            'service_type': service_type,
                            'distance': distance,
                            'city': city,
                            'hour': hour,
                            'day_of_week': day_of_week,
                            'sanity_warnings': sanity_warnings
                        }
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    import traceback
                    with st.expander("See detailed error"):
                        st.code(traceback.format_exc())
                    st.info("This usually happens when the model wasn't trained with all possible service types or cities.")
        
        # Display prediction result OUTSIDE the form but INSIDE the column
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            
            # Show raw vs adjusted prediction if they differ
            if abs(result['prediction'] - result.get('raw_prediction', result['prediction'])) > 0.1:
                st.info(f"📝 Note: Prediction adjusted from {result.get('raw_prediction', result['prediction']):.2f}h based on sanity checks")
            
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #00b894 0%, #00a382 100%);
                color: white;
                padding: 2rem;
                border-radius: 0.5rem;
                text-align: center;
                margin: 1rem 0;
            '>
                <h2 style='margin: 0; font-size: 2.5rem;'>📦 {result['prediction']:.2f} hours</h2>
                <p style='margin: 0.5rem 0 0 0;'>Predicted Delivery Time</p>
                <p style='margin: 0.2rem 0; font-size: 0.9rem;'>
                    For: {result['distance']:.1f}km to {result['city']} ({result['service_type']})
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional context and recommendations
            if result['prediction'] < 2:
                st.success("⚡ **Express-level delivery**: This is faster than 90% of deliveries!")
            elif result['prediction'] < 4:
                st.info("🚗 **Standard delivery**: Typical delivery time for this distance")
            elif result['prediction'] < 8:
                st.warning("🐢 **Extended delivery**: Consider express service for urgent packages")
            else:
                st.error("⏳ **Long delivery**: This route may have operational challenges")
            
            # Service recommendation
            if result['prediction'] > 6 and result['service_type'] != "EXPRESS":
                st.info("💡 **Recommendation**: Consider EXPRESS service for faster delivery")
            
            # Show sanity warnings if any
            if result.get('sanity_warnings'):
                with st.expander("⚠️ Prediction Notes"):
                    for warning in result['sanity_warnings']:
                        st.write(f"• {warning}")
    
    # Data Quality Assessment
    st.markdown("---")
    st.markdown("### 🛠️ Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Common Data Issues:**")
        st.write("• Hashed service type names")
        st.write("• Inconsistent city naming")
        st.write("• Missing or outlier delivery times")
        st.write("• Incorrect distance calculations")
    
    with col2:
        st.write("**Recommended Fixes:**")
        st.write("• Check raw data for readable service types")
        st.write("• Verify city name consistency")
        st.write("• Clean delivery time outliers")
        st.write("• Review distance calculation method")
    
    # Debug information
    st.markdown("---")
    with st.expander("🔧 Debug Information"):
        st.write("**Model Details:**")
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            st.write(f"- Model type: {type(st.session_state.model).__name__}")
            st.write(f"- Expected features: {st.session_state.model.n_features_in_}")
            
            # Try to get feature names if available
            try:
                if hasattr(st.session_state.model, 'feature_names_in_'):
                    st.write("- Feature names:", list(st.session_state.model.feature_names_in_))
            except:
                pass
        
        st.write("**Data Sample:**")
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            sample_df = st.session_state.df.head(3)
            st.dataframe(sample_df[['from_city_name', 'service_type', 'distance_km', 'delivery_time_hours']] 
                        if all(col in sample_df.columns for col in ['from_city_name', 'service_type', 'distance_km', 'delivery_time_hours']) 
                        else sample_df)
        
        # Prediction history
        if hasattr(st.session_state, 'prediction_result') and st.session_state.prediction_result:
            result = st.session_state.prediction_result
            st.write("**Last Prediction Details:**")
            st.write(f"- Distance: {result['distance']} km")
            st.write(f"- Service Type: {result['service_type']}")
            st.write(f"- City: {result['city']}")
            st.write(f"- Hour: {result['hour']}")
            st.write(f"- Day: {result['day_of_week']}")
            st.write(f"- Raw Prediction: {result.get('raw_prediction', result['prediction']):.2f} hours")
            st.write(f"- Final Prediction: {result['prediction']:.2f} hours")
            
            if result.get('sanity_warnings'):
                st.write("- Sanity Warnings:")
                for warning in result['sanity_warnings']:
                    st.write(f"  • {warning}")
                                            
def show_anomaly_section(df, anomalies, anomaly_by_city, anomaly_by_type):
    st.markdown('<div class="main-header">Anomaly Detection</div>', unsafe_allow_html=True)
    
    anomaly_count = (anomalies == -1).sum() if hasattr(anomalies, 'sum') else 0
    st.metric("Detected Anomalies", anomaly_count)
    
    # Show top anomalies
    if hasattr(anomalies, 'sum') and anomaly_count > 0:
        anomaly_df = df[anomalies == -1].head(5)
        for _, row in anomaly_df.iterrows():
            st.warning(f"🚨 Anomaly detected: Order {row['order_id']} took {row['delivery_time_hours']:.2f} hours")
    
    # User anomaly report form
    st.markdown("---")
    st.subheader("Report Suspicious Delivery")
    
    with st.form("anomaly_report_form"):
        st.write("### Report a delivery that seems suspicious")
        
        order_id = st.text_input("Order ID", placeholder="e.g., ORD_123456")
        
        col1, col2 = st.columns(2)
        with col1:
            actual_time = st.number_input("Actual Delivery Time (hours)", min_value=0.1, max_value=168.0, value=2.0)
        with col2:
            expected_time = st.number_input("Expected Delivery Time (hours)", min_value=0.1, max_value=168.0, value=1.5)
        
        reason = st.selectbox(
            "Why does this seem suspicious?",
            options=["Much longer than expected", "Much shorter than expected", 
                    "Different route than usual", "Other reason"]
        )
        
        description = st.text_area("Additional details", placeholder="Describe what seemed unusual...")
        
        reported = st.form_submit_button("Report Anomaly")
        
        if reported:
            if actual_time > expected_time * 2:
                st.error("🚨 This delivery took more than twice the expected time! Flagged for review.")
            elif actual_time < expected_time * 0.5:
                st.error("🚨 This delivery was completed in half the expected time! Flagged for review.")
            else:
                st.info("📋 Thank you for your report. We'll monitor this delivery pattern.")
            
            # In a real system, you would save this to a database
            st.success("Report submitted successfully! Our team will investigate.")

    from src.visualization import create_anomaly_visualizations
    st.plotly_chart(create_anomaly_visualizations(df, anomalies), use_container_width=True)

# Add this function to your app.py
def show_data_upload_section():
    st.markdown('<div class="main-header">Upload Delivery Data</div>', unsafe_allow_html=True)
    
    st.info("📤 Upload new delivery data for processing and analysis")
    
    tab1, tab2 = st.tabs(["Upload CSV File", "Manual Entry"])
    
    with tab1:
        st.write("### Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                new_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully uploaded {len(new_data)} records")
                
                # Show preview
                st.write("### Data Preview")
                st.dataframe(new_data.head())
                
                # Process button
                if st.button("Process Uploaded Data"):
                    with st.spinner("Processing data..."):
                        # Here you would integrate this with your processing pipeline
                        st.success("Data processed successfully!")
                        st.info("The new data has been added to your analytics dashboard.")
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        st.write("### Manual Data Entry")
        
        with st.form("manual_entry_form"):
            st.write("Enter delivery details manually")
            
            col1, col2 = st.columns(2)
            with col1:
                order_id = st.text_input("Order ID")
                from_city = st.selectbox("From City", ["Dar es Salaam", "Arusha", "Mwanza", "Mbeya", "Dodoma"])
                distance = st.number_input("Distance (km)", min_value=0.1, value=5.0)
            with col2:
                service_type = st.selectbox("Service Type", ["EXPRESS", "STANDARD", "ECONOMY"])
                delivery_time = st.number_input("Delivery Time (hours)", min_value=0.1, value=2.0)
                received_time = st.time_input("Time Received")
            
            submitted = st.form_submit_button("Add Delivery Record")
            
            if submitted:
                # In a real system, you'd add this to your database
                st.success("Delivery record added successfully!")


def show_basic_map(df):
    """Show basic delivery map with MORE data points"""
    try:
        # Use MUCH larger sample size - up to 10,000 points!
        if len(df) > 10000:
            map_df = df.sample(10000, random_state=42)  # 10,000 points!
            st.info(f"📊 Showing 10,000 random delivery points (sampled from {len(df):,} total)")
        elif len(df) > 5000:
            map_df = df.sample(5000, random_state=42)  # 5,000 points
            st.info(f"📊 Showing 5,000 random delivery points")
        elif len(df) > 1000:
            map_df = df.sample(1000, random_state=42)  # 1,000 points
            st.info(f"📊 Showing 1,000 random delivery points")
        else:
            map_df = df.copy()
            st.info(f"📊 Showing all {len(df)} delivery points")
        
        # Create map data with proper coordinates
        map_data = pd.DataFrame()
        
        # Try multiple coordinate sources
        coordinate_sources = [
            ('receipt_lat', 'receipt_lng'),
            ('poi_lat', 'poi_lng'),
            ('from_city_lat', 'from_city_lng'),
            ('to_city_lat', 'to_city_lng')
        ]
        
        # Collect coordinates from all available sources
        all_coords = []
        
        for lat_col, lng_col in coordinate_sources:
            if all(col in map_df.columns for col in [lat_col, lng_col]):
                valid_coords = map_df[[lat_col, lng_col]].dropna()
                if len(valid_coords) > 0:
                    # Add coordinates to our collection
                    temp_df = pd.DataFrame()
                    temp_df['lat'] = valid_coords[lat_col]
                    temp_df['lon'] = valid_coords[lng_col]
                    
                    # Add additional information if available
                    if 'from_city_name' in map_df.columns:
                        temp_df['city'] = map_df.loc[valid_coords.index, 'from_city_name'].values
                    
                    if 'delivery_time_hours' in map_df.columns:
                        temp_df['delivery_time'] = map_df.loc[valid_coords.index, 'delivery_time_hours'].values
                    
                    all_coords.append(temp_df)
        
        # Combine all coordinate sources
        if all_coords:
            map_data = pd.concat(all_coords, ignore_index=True)
            
            # Remove duplicates (same coordinates)
            map_data = map_data.drop_duplicates(subset=['lat', 'lon'])
            
            # Add size based on delivery time if available
            if 'delivery_time' in map_data.columns:
                map_data['size'] = np.clip(map_data['delivery_time'] / 2, 5, 20)
            else:
                map_data['size'] = 10  # Default size
        
        # Remove any rows with missing coordinates
        map_data = map_data.dropna(subset=['lat', 'lon'])
        
        if len(map_data) == 0:
            st.warning("No valid coordinates available for mapping")
            # Show debug information
            st.write("**Debug Info:**")
            st.write(f"Total records: {len(df)}")
            
            # Check which coordinate columns exist
            for lat_col, lng_col in coordinate_sources:
                if lat_col in df.columns and lng_col in df.columns:
                    valid_count = df[df[lat_col].notna() & df[lng_col].notna()].shape[0]
                    st.write(f"Valid {lat_col}/{lng_col} coordinates: {valid_count}")
            
            return
        
        # Display the map with MORE points
        st.map(map_data, use_container_width=True, color='#1f77b4', size='size')
        
        # Show map stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📍 Locations Mapped", f"{len(map_data):,}")
        with col2:
            if 'delivery_time' in map_data.columns:
                st.metric("⏱️ Avg Delivery Time", f"{map_data['delivery_time'].mean():.1f} hrs")
            else:
                st.metric("📊 Points Displayed", f"{len(map_data):,}")
        with col3:
            if 'city' in map_data.columns:
                st.metric("🏙️ Cities Covered", f"{map_data['city'].nunique()}")
            
    except Exception as e:
        st.error(f"Map visualization error: {str(e)}")
        import traceback
        st.write("Error details:", traceback.format_exc())
        
def show_delivery_map(df):
    """Dedicated map page with MUCH more data points"""
    st.markdown('<div class="main-header">🌍 Delivery Operations Map</div>', unsafe_allow_html=True)
    
    # Show debug info first
    show_map_debug_info(df)
    
    # Map type selector
    map_type = st.radio("Map View", ["Basic Map", "Advanced Visualization", "City Performance", "Heatmap"], horizontal=True)
    
    # Ensure we have enough data for visualization
    if len(df) < 10:
        st.warning("Not enough data points for map visualization")
        return
    
    # Use MUCH larger sample sizes for better visualization
    if map_type == "Basic Map":
        show_basic_map(df)  # No sampling - let the function handle it
    elif map_type == "Advanced Visualization":
        show_advanced_map(df)
    elif map_type == "City Performance":
        show_city_map(df)
    else:
        show_heatmap(df)
        
def show_advanced_map(df):
    """Show advanced Plotly map with better sampling"""
    try:
        # Sample more points for advanced visualization
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        from src.visualization import create_advanced_delivery_map
        fig = create_advanced_delivery_map(sample_df, sample_size=1000)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("🎯 **Advanced Map Features:**\n- Color indicates delivery time (darker = longer)\n- Size indicates delivery time (larger = longer)\n- Hover for detailed information")
        
    except Exception as e:
        st.error(f"Advanced map error: {str(e)}")

# Add this to your main map function or sidebar
def show_map_debug_info(df):
    """Show debug information for map data"""
    with st.expander("🗺️ Map Debug Information"):
        st.write(f"Total dataset records: {len(df):,}")
        
        if 'receipt_lat' in df.columns and 'receipt_lng' in df.columns:
            valid_receipt = df[df['receipt_lat'].notna() & df['receipt_lng'].notna()]
            st.write(f"Valid receipt coordinates: {len(valid_receipt):,}")
            
        if 'poi_lat' in df.columns and 'poi_lng' in df.columns:
            valid_poi = df[df['poi_lat'].notna() & df['poi_lng'].notna()]
            st.write(f"Valid POI coordinates: {len(valid_poi):,}")
            
        if 'from_city_name' in df.columns:
            st.write(f"Unique cities: {df['from_city_name'].nunique()}")
            st.write("City distribution:")
            st.write(df['from_city_name'].value_counts().head(10))


def show_city_map(df):
    """Show city performance map"""
    try:
        from src.visualization import create_city_delivery_map
        fig = create_city_delivery_map(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("🏙️ **City Performance Map:**\n- Color indicates average delivery time\n- Size indicates number of deliveries\n- Compare performance across cities")
        
    except Exception as e:
        st.error(f"City map error: {str(e)}")

def show_heatmap(df):
    """Show delivery density heatmap"""
    try:
        from src.visualization import create_heatmap
        import streamlit.components.v1 as components
        
        st.info("🔥 **Heatmap shows delivery density:**\n- Red areas: High concentration of deliveries\n- Blue areas: Fewer deliveries\n- Useful for identifying service hotspots")
        
        # Create heatmap
        m = create_heatmap(df, sample_size=2000)
        
        # Save to HTML and display
        m.save('heatmap.html')
        components.html(open('heatmap.html', 'r').read(), height=500)
        
    except ImportError:
        st.warning("Heatmap requires folium package. Install with: `pip install folium`")
    except Exception as e:
        st.error(f"Heatmap error: {str(e)}")

def show_anomaly_section(df, anomalies, anomaly_by_city, anomaly_by_type):
    st.markdown('<div class="main-header">Anomaly Detection</div>', unsafe_allow_html=True)
    
    anomaly_count = (anomalies == -1).sum() if hasattr(anomalies, 'sum') else 0
    st.metric("Detected Anomalies", anomaly_count)
    
    # Show top anomalies
    if hasattr(anomalies, 'sum') and anomaly_count > 0:
        anomaly_df = df[anomalies == -1].head(5)
        for _, row in anomaly_df.iterrows():
            st.warning(f"Anomaly detected: Order {row['order_id']} took {row['delivery_time_hours']:.2f} hours")
    
    from src.visualization import create_anomaly_visualizations
    st.plotly_chart(create_anomaly_visualizations(df, anomalies), use_container_width=True)

def show_insights_section(df):
    st.markdown('<div class="main-header">Data Insights</div>', unsafe_allow_html=True)
    st.dataframe(df.describe())
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    import plotly.express as px
    fig = px.imshow(corr_matrix, title='Feature Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

def load_and_process_data(sample_size):
    """Load and process data with proper error handling"""
    with st.spinner("Loading and processing data (this may take a few minutes)..."):
        try:
            if not os.path.exists("data/courier_dataset.csv"):
                st.warning("📁 Data file not found. Using synthetic data for demonstration.")
                from src.utils import generate_synthetic_data
                st.session_state.df = generate_synthetic_data(10000)
            else:
                from src.data_processing import load_data_with_cache
                st.session_state.df = load_data_with_cache("data/courier_dataset.csv", sample_size=sample_size)
                st.success(f"✅ Dataset loaded successfully! ({len(st.session_state.df):,} records)")
            
            # Prepare features
            from src.data_processing import prepare_features_with_cache, prepare_anomaly_features_with_cache
            st.session_state.X, st.session_state.y, st.session_state.le_city, st.session_state.le_type = prepare_features_with_cache(
                st.session_state.df, sample_size=50000
            )
            st.session_state.X_anomaly_scaled, st.session_state.scaler, st.session_state.df_anomaly = prepare_anomaly_features_with_cache(
                st.session_state.df, sample_size=100000
            )
            
            st.session_state.data_loaded = True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            from src.utils import generate_synthetic_data
            from src.data_processing import prepare_features_with_cache, prepare_anomaly_features_with_cache
            st.session_state.df = generate_synthetic_data(10000)
            st.session_state.X, st.session_state.y, st.session_state.le_city, st.session_state.le_type = prepare_features_with_cache(st.session_state.df)
            st.session_state.X_anomaly_scaled, st.session_state.scaler, st.session_state.df_anomaly = prepare_anomaly_features_with_cache(st.session_state.df)
            st.session_state.data_loaded = True

def process_app_mode(app_mode):
    """Process the selected application mode"""
    with st.spinner("Training machine learning models..."):
        try:
            from src.model_training import train_delivery_time_model, analyze_model_performance
            
            # Train model if not already trained
            if st.session_state.model is None:
                (st.session_state.model, X_test, y_test, y_pred, 
                 st.session_state.mae, st.session_state.rmse, st.session_state.r2) = train_delivery_time_model(
                    st.session_state.X, st.session_state.y
                )
                
                # Detailed performance analysis
                performance = analyze_model_performance(y_test, y_pred, "Delivery Time Prediction")
                st.session_state.performance = performance
            
            st.success("✅ Models trained successfully!")
            
            # Route to appropriate section
            if app_mode == "Dashboard":
                show_dashboard_section()
            elif app_mode == "Delivery Time Prediction":
                show_prediction_section()
            elif app_mode == "Anomaly Detection":
                show_anomaly_detection_section()
            elif app_mode == "Data Insights":
                show_insights_section(st.session_state.df)
            elif app_mode == "Delivery Map":
                show_delivery_map(st.session_state.df)
            elif app_mode == "Upload Data":
                show_data_upload_section()
                
        except Exception as e:
            st.error(f"❌ Error in application: {str(e)}")
            import traceback
            with st.expander("See error details"):
                st.code(traceback.format_exc())

def show_dashboard_section():
    """Display dashboard with comprehensive analytics"""
    from src.visualization import create_realistic_delivery_plot, create_operational_anomalies_plot, create_speed_analysis_dashboard
    
    show_dashboard(st.session_state.df, None)
    
    # Realistic Delivery Analysis
    st.markdown("---")
    st.markdown('<div class="custom-header">Delivery Intelligence Dashboard</div>', unsafe_allow_html=True)
    realistic_fig, df_realistic = create_realistic_delivery_plot(st.session_state.df)
    st.plotly_chart(realistic_fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    total_deliveries = len(st.session_state.df)
    realistic_deliveries = len(df_realistic)
    
    with col1:
        st.metric("Total Deliveries", total_deliveries)
    with col2:
        st.metric("Realistic Deliveries", f"{realistic_deliveries} ({realistic_deliveries/total_deliveries*100:.1f}%)")
    with col3:
        invalid = total_deliveries - realistic_deliveries
        st.metric("Data Errors", f"{invalid} ({invalid/total_deliveries*100:.1f}%)")
    with col4:
        problems = len(df_realistic[df_realistic['operational_anomaly'] != 'normal']) if 'operational_anomaly' in df_realistic.columns else 0
        st.metric("Operational Problems", f"{problems} ({problems/realistic_deliveries*100:.1f}%)" if realistic_deliveries > 0 else "0")
    
    # Operational Anomalies
    st.markdown("## 🔍 Operational Anomalies Needing Investigation")
    anomalies_fig = create_operational_anomalies_plot(df_realistic)
    st.plotly_chart(anomalies_fig, use_container_width=True)
    
    # Speed Analysis
    st.markdown("## 📊 Speed Analysis Dashboard")
    speed_fig = create_speed_analysis_dashboard(st.session_state.df)
    st.plotly_chart(speed_fig, use_container_width=True)

def debug_model_performance():
    """Debug why model performance is poor"""
    st.markdown("### 🔍 Model Performance Debug")
    
    if hasattr(st.session_state, 'X') and hasattr(st.session_state, 'y'):
        X, y = st.session_state.X, st.session_state.y
        
        # Check for data issues
        st.write("**Data Analysis:**")
        st.write(f"• Samples: {X.shape[0]}")
        st.write(f"• Features: {X.shape[1]}")
        st.write(f"• Target mean: {y.mean():.2f}")
        st.write(f"• Target std: {y.std():.2f}")
        
        # Check for constant features
        constant_features = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) < 0.001:  # Almost constant
                constant_features.append(i)
        
        if constant_features:
            st.error(f"❌ Found {len(constant_features)} constant/near-constant features!")
        
        # Check correlation with target
        if hasattr(st.session_state, 'model') and hasattr(st.session_state.model, 'feature_importances_'):
            st.write("**Feature Importance (Top 10):**")
            importances = st.session_state.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(10, len(importances))):
                st.write(f"  Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        # Check if target has meaningful variance
        if y.std() < 0.1:
            st.error("❌ Target variable has very low variance!")

def debug_data_quality():
    """Debug function to inspect data quality issues"""
    st.markdown("### 🔍 Data Quality Inspection")
    
    if hasattr(st.session_state, 'df'):
        df = st.session_state.df
        
        # Check service types
        st.write("**Service Types Analysis:**")
        if 'typecode' in df.columns:
            service_types = df['typecode'].dropna().unique()
            st.write(f"Found {len(service_types)} unique service types:")
            
            # Show service types with their frequencies
            service_counts = df['typecode'].value_counts().head(20)
            st.write("Most common service types:")
            for service, count in service_counts.items():
                st.write(f"  '{service}': {count} records")
            
            # Check for hashed values
            hashed_services = [stype for stype in service_types if isinstance(stype, str) and len(stype) > 20 and stype.isalnum()]
            if hashed_services:
                st.error(f"❌ Found {len(hashed_services)} potentially hashed service types!")
                st.write("Sample hashed values:", hashed_services[:3])
        
        # Check cities
        st.write("**Cities Analysis:**")
        if 'from_city_name' in df.columns:
            cities = df['from_city_name'].dropna().unique()
            st.write(f"Found {len(cities)} unique cities:")
            
            city_counts = df['from_city_name'].value_counts().head(10)
            for city, count in city_counts.items():
                st.write(f"  '{city}': {count} records")
        
        # Check delivery times
        st.write("**Delivery Time Analysis:**")
        if 'delivery_time_hours' in df.columns:
            delivery_times = df['delivery_time_hours']
            st.write(f"• Mean: {delivery_times.mean():.2f} hours")
            st.write(f"• Median: {delivery_times.median():.2f} hours")
            st.write(f"• Std Dev: {delivery_times.std():.2f} hours")
            st.write(f"• Min: {delivery_times.min():.2f} hours")
            st.write(f"• Max: {delivery_times.max():.2f} hours")
            
            # Check for outliers
            Q1 = delivery_times.quantile(0.25)
            Q3 = delivery_times.quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(delivery_times < (Q1 - 1.5 * IQR)) | (delivery_times > (Q3 + 1.5 * IQR))]
            st.write(f"• Statistical outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.1f}%)")
            
            # Check for impossible values
            impossible = df[(delivery_times <= 0) | (delivery_times > 168)]  # More than 1 week
            st.write(f"• Impossible values: {len(impossible):,} ({len(impossible)/len(df)*100:.1f}%)")
        
        # Check distances
        st.write("**Distance Analysis:**")
        if 'distance_km' in df.columns:
            distances = df['distance_km']
            st.write(f"• Mean: {distances.mean():.2f} km")
            st.write(f"• Median: {distances.median():.2f} km")
            st.write(f"• Min: {distances.min():.2f} km")
            st.write(f"• Max: {distances.max():.2f} km")
            
            # Check for impossible distances
            impossible_dist = df[(distances <= 0) | (distances > 1000)]  # More than 1000km
            st.write(f"• Impossible distances: {len(impossible_dist):,} ({len(impossible_dist)/len(df)*100:.1f}%)")


def show_anomaly_detection_section():
    """Display anomaly detection section with improved visualizations"""
    from src.anomaly_detection import detect_anomalies, analyze_anomalies
    from src.visualization import create_operational_anomalies_plot, create_realistic_delivery_plot
    
    # Compute anomalies if not already done
    if st.session_state.anomalies is None:
        with st.spinner("Detecting anomalies..."):
            st.session_state.anomalies = detect_anomalies(st.session_state.X_anomaly_scaled)
            st.session_state.df_with_anomalies, st.session_state.anomaly_by_city, st.session_state.anomaly_by_type = analyze_anomalies(
                st.session_state.df_anomaly, st.session_state.anomalies
            )
    
    # Show main anomaly detection section
    st.markdown('<div class="main-header">Anomaly Detection</div>', unsafe_allow_html=True)
    
    anomaly_count = (st.session_state.anomalies == -1).sum() if hasattr(st.session_state.anomalies, 'sum') else 0
    st.metric("🚨 Detected Anomalies", anomaly_count)
    
    # Show top anomalies
    if hasattr(st.session_state.anomalies, 'sum') and anomaly_count > 0:
        st.markdown("### 🔍 Recent Anomalies Detected")
        anomaly_df = st.session_state.df_with_anomalies[st.session_state.anomalies == -1].head(5)
        for _, row in anomaly_df.iterrows():
            st.markdown(f"""
            <div class="anomaly-alert">
                <strong>Order {row['order_id']}</strong><br>
                📍 {row['from_city_name']} | 🚚 {row['typecode']}<br>
                ⏱️ {row['delivery_time_hours']:.2f} hours | 📏 {row['distance_km']:.2f} km<br>
                🕒 {row['receipt_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['receipt_time'], 'strftime') else 'Unknown time'}
            </div>
            """, unsafe_allow_html=True)
    
    # User anomaly report form
    st.markdown("---")
    st.subheader("📝 Report Suspicious Delivery")
    
    with st.form("anomaly_report_form"):
        st.markdown("### Report a delivery that seems unusual or suspicious")
        
        order_id = st.text_input("Order ID", placeholder="e.g., ORD_123456", help="Enter the order ID if available")
        
        col1, col2 = st.columns(2)
        with col1:
            actual_time = st.number_input("Actual Delivery Time (hours)", min_value=0.1, max_value=168.0, value=2.0, step=0.1)
        with col2:
            expected_time = st.number_input("Expected Delivery Time (hours)", min_value=0.1, max_value=168.0, value=1.5, step=0.1)
        
        reason = st.selectbox(
            "Why does this seem suspicious?",
            options=["Much longer than expected", "Much shorter than expected", 
                    "Different route than usual", "Package condition issues", "Other reason"]
        )
        
        description = st.text_area("Additional details", placeholder="Describe what seemed unusual about this delivery...", height=100)
        
        reported = st.form_submit_button("📤 Submit Report")
        
        if reported:
            if actual_time > expected_time * 2:
                st.error("🚨 This delivery took more than twice the expected time! Flagged for urgent review.")
            elif actual_time < expected_time * 0.5:
                st.error("🚨 This delivery was completed in half the expected time! Flagged for investigation.")
            else:
                st.info("📋 Thank you for your report. We'll monitor this delivery pattern.")
            
            st.success("✅ Report submitted successfully! Our operations team will investigate.")

    # Operational Anomalies Section with Improved Visualization
    st.markdown("---")
    st.markdown("## 🚨 Operational Anomalies Analysis")
    
    # Create realistic delivery plot and get filtered data
    realistic_fig, df_realistic = create_realistic_delivery_plot(st.session_state.df_with_anomalies)
    st.plotly_chart(realistic_fig, use_container_width=True)
    
    # Filter only problematic deliveries
    if 'operational_anomaly' in df_realistic.columns:
        df_problems = df_realistic[df_realistic['operational_anomaly'] != 'normal'].copy()
        
        if len(df_problems) > 0:
            # Summary statistics
            st.markdown("### 📊 Anomaly Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_speed = df_problems['speed_kmh_calc'].mean()
                st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
            
            with col2:
                avg_time = df_problems['delivery_time_hours'].mean()
                st.metric("Avg Time", f"{avg_time:.1f} hours")
            
            with col3:
                avg_distance = df_problems['distance_km'].mean()
                st.metric("Avg Distance", f"{avg_distance:.1f} km")
            
            with col4:
                total_anomalies = len(df_problems)
                st.metric("Total Cases", total_anomalies)
            
            # Anomaly type breakdown
            st.markdown("### 🔍 Anomaly Type Breakdown")
            anomaly_counts = df_problems['operational_anomaly'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of anomaly types
                fig_pie = px.pie(
                    values=anomaly_counts.values,
                    names=anomaly_counts.index,
                    title="Distribution of Anomaly Types",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart of anomaly counts
                fig_bar = px.bar(
                    x=anomaly_counts.index,
                    y=anomaly_counts.values,
                    title="Anomaly Count by Type",
                    labels={'x': 'Anomaly Type', 'y': 'Count'},
                    color=anomaly_counts.index,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Enhanced operational anomalies plot (without walking pace reference)
            st.markdown("### 📈 Detailed Anomaly Analysis")
            operational_fig = create_operational_anomalies_plot(df_realistic)
            st.plotly_chart(operational_fig, use_container_width=True)
            
            # Anomaly details table
            st.markdown("### 📋 Anomaly Details")
            
            # Define columns that actually exist in your dataset
            available_columns = []
            possible_columns = [
                'order_id', 'from_city_name', 'distance_km', 
                'delivery_time_hours', 'speed_kmh_calc', 'operational_anomaly',
                'receipt_time', 'typecode'
            ]
            
            for col in possible_columns:
                if col in df_problems.columns:
                    available_columns.append(col)
            
            # Show the top 10 most severe anomalies (longest delivery times)
            top_anomalies = df_problems[available_columns].sort_values(
                'delivery_time_hours', ascending=False
            ).head(10)
            
            st.dataframe(top_anomalies, use_container_width=True)
            
            # Download button for anomaly data
            csv = df_problems[available_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download Anomaly Data (CSV)",
                data=csv,
                file_name="operational_anomalies.csv",
                mime="text/csv",
                help="Download detailed information about all operational anomalies"
            )
            
        else:
            st.success("🎉 No operational anomalies detected! All deliveries appear to be operating normally.")
    else:
        st.info("ℹ️ Operational anomaly detection is not available in the current analysis.")

def main():
    """
    Main application function
    """
    import os

    # Load dataset
    data_path = "data/courier_dataset.csv"
    if not os.path.exists(data_path):
        st.error("Dataset not found. Please run `python download_data.py` or check Git LFS.")
        return
    
    try:
        df = pd.read_csv(data_path)
        max_rows = len(df) if not df.empty else 1  # Fallback if empty
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Ensure valid max_rows
    if max_rows < 10000:
        st.warning(f"Dataset has only {max_rows} records, which is less than the minimum sample size (10,000). Using full dataset.")
        sample_size = max_rows
    else:
        # Define slider with safe parameters
        use_sample = st.sidebar.checkbox("Use sample data", value=True)
        if use_sample:
            sample_size = st.sidebar.slider(
                label="Sample size",
                min_value=10000,
                max_value=max_rows,
                value=min(50000, max_rows),
                step=1000,
                help=f"Select number of records to use (max: {max_rows:,})"
            )
        else:
            sample_size = max_rows

    # Use the sampled dataset
    sampled_df = df.sample(n=sample_size, random_state=42) if use_sample else df
    st.write(f"Dataset size: {sample_size} records")

    # Load model
    model_path = "models/delivery_time_model.pkl"
    if not os.path.exists(model_path):
        st.info("Regenerating model...")
        os.system("python src/model_training.py")
    model = joblib.load(model_path)

    # Initialize session state to prevent double loading
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.X = None
        st.session_state.y = None
        st.session_state.le_city = None
        st.session_state.le_type = None
        st.session_state.feature_names = None
        st.session_state.X_anomaly_scaled = None
        st.session_state.scaler = None
        st.session_state.df_anomaly = None
        st.session_state.model = None
        st.session_state.mae = None
        st.session_state.rmse = None
        st.session_state.r2 = None
        st.session_state.anomalies = None
        st.session_state.df_with_anomalies = None
        st.session_state.anomaly_by_city = None
        st.session_state.anomaly_by_type = None
        st.session_state.performance = None
    
    st.title("📦 AI-Powered Delivery Intelligence System")
    st.markdown("*Smart logistics powered by machine learning*")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    use_sample = st.sidebar.checkbox("Use sample data for faster processing", value=True)
    
    # Determine maximum dataset size
    max_rows = 500000  # Default maximum
    
    # Try to detect actual file size if the file exists
    try:
        if os.path.exists("data/courier_dataset.csv"):
            with open("data/courier_dataset.csv", 'r', encoding='utf-8') as f:
                line_count = sum(1 for line in f)
            max_rows = min(line_count - 1, 500000)
            st.sidebar.success(f"📊 Dataset size: {max_rows:,} records")
    except Exception as e:
        st.sidebar.warning(f"Could not determine file size: {e}")
        max_rows = 500000
    
    # Sample size selection
    if use_sample:
        sample_size = st.sidebar.slider(
            "Sample size", 
            min_value=10000, 
            max_value=max_rows, 
            value=min(50000, max_rows),
            step=1000,
            help=f"Select number of records to use (max: {max_rows:,})"
        )
        st.sidebar.info(f"Using {sample_size:,} records")
    else:
        sample_size = None
        st.sidebar.info("Using full dataset")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data (Clear Cache)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.success("Cache cleared! Refreshing...")
        st.rerun()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", 
                                   ["Dashboard", "Delivery Time Prediction", 
                                    "Anomaly Detection", "Data Insights", 
                                    "Delivery Map", "Upload Data"])
    
    # Load data ONLY ONCE using session state
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data (this may take a few minutes)..."):
            try:
                # Check if data file exists
                if not os.path.exists("data/courier_dataset.csv"):
                    st.warning("📁 Data file not found. Using synthetic data for demonstration.")
                    from src.utils import generate_synthetic_data
                    st.session_state.df = generate_synthetic_data(10000)
                else:
                    # Use cached data loading
                    from src.data_processing import load_data_with_cache, decode_hashed_service_types
                    st.session_state.df = load_data_with_cache("data/courier_dataset.csv", sample_size=sample_size)
                    
                    # Check if dataframe is valid
                    if st.session_state.df is None or len(st.session_state.df) == 0:
                        st.error("❌ Failed to load data. Please check your data file.")
                        return
                    
                    st.success(f"✅ Dataset loaded successfully! ({len(st.session_state.df):,} records)")
                
                # DECODE HASHED SERVICE TYPES FIRST
                st.info("🔍 Decoding hashed service types...")
                st.session_state.df = decode_hashed_service_types(st.session_state.df)
                
                # Show service type distribution
                if 'service_type' in st.session_state.df.columns:
                    service_counts = st.session_state.df['service_type'].value_counts()
                    st.write("**Service Type Distribution:**")
                    for service, count in service_counts.items():
                        st.write(f"• {service}: {count} records")
                
                # Prepare features with enhanced function
                from src.data_processing import prepare_enhanced_features
                (st.session_state.X, st.session_state.y, st.session_state.le_city, 
                 st.session_state.le_type, st.session_state.feature_names) = prepare_enhanced_features(
                    st.session_state.df, sample_size=50000
                )
                
                # Prepare anomaly features
                from src.data_processing import prepare_anomaly_features_with_cache
                (st.session_state.X_anomaly_scaled, st.session_state.scaler, 
                 st.session_state.df_anomaly) = prepare_anomaly_features_with_cache(
                    st.session_state.df, sample_size=100000
                )
                
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())
                
                # Fallback to synthetic data
                from src.utils import generate_synthetic_data
                from src.data_processing import prepare_enhanced_features, prepare_anomaly_features_with_cache, decode_hashed_service_types
                
                st.session_state.df = generate_synthetic_data(10000)
                st.session_state.df = decode_hashed_service_types(st.session_state.df)
                
                (st.session_state.X, st.session_state.y, st.session_state.le_city, 
                 st.session_state.le_type, st.session_state.feature_names) = prepare_enhanced_features(st.session_state.df)
                
                (st.session_state.X_anomaly_scaled, st.session_state.scaler, 
                 st.session_state.df_anomaly) = prepare_anomaly_features_with_cache(st.session_state.df)
                
                st.session_state.data_loaded = True
    else:
        st.success(f"✅ Using cached data ({len(st.session_state.df):,} records)")
    
    # Process based on selected mode
    if st.session_state.data_loaded and st.session_state.df is not None:
        with st.spinner("Processing your request..."):
            try:
                from src.model_training import train_enhanced_delivery_time_model, analyze_model_performance
                
                # Train model if not already trained
                if st.session_state.model is None:
                    (st.session_state.model, X_test, y_test, y_pred, 
                     st.session_state.mae, st.session_state.rmse, st.session_state.r2) = train_enhanced_delivery_time_model(
                        st.session_state.X, st.session_state.y
                    )
                    
                    # Detailed performance analysis
                    performance = analyze_model_performance(y_test, y_pred, "Delivery Time Prediction")
                    st.session_state.performance = performance
                
                st.success("✅ Models trained successfully!")
                
                # Display appropriate section
                if app_mode == "Dashboard":
                    from src.visualization import create_realistic_delivery_plot, create_operational_anomalies_plot, create_speed_analysis_dashboard
                    
                    show_dashboard(st.session_state.df, None)
                    
                    # Realistic Delivery Analysis
                    st.markdown("---")
                    st.markdown('<div class="custom-header">🚀 Realistic Delivery Analysis</div>', unsafe_allow_html=True)
                    
                    realistic_fig, df_realistic = create_realistic_delivery_plot(st.session_state.df)
                    st.plotly_chart(realistic_fig, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    total_deliveries = len(st.session_state.df)
                    realistic_deliveries = len(df_realistic)
                    
                    with col1:
                        st.metric("Total Deliveries", total_deliveries)
                    with col2:
                        st.metric("Realistic Deliveries", f"{realistic_deliveries} ({realistic_deliveries/total_deliveries*100:.1f}%)")
                    with col3:
                        invalid = total_deliveries - realistic_deliveries
                        st.metric("Data Errors", f"{invalid} ({invalid/total_deliveries*100:.1f}%)")
                    with col4:
                        problems = len(df_realistic[df_realistic['operational_anomaly'] != 'normal']) if 'operational_anomaly' in df_realistic.columns else 0
                        st.metric("Operational Problems", f"{problems} ({problems/realistic_deliveries*100:.1f}%)" if realistic_deliveries > 0 else "0")
                    
                    # Operational Anomalies
                    st.markdown("## 🔍 Operational Anomalies Needing Investigation")
                    anomalies_fig = create_operational_anomalies_plot(df_realistic)
                    st.plotly_chart(anomalies_fig, use_container_width=True)
                    
                    # Speed Analysis
                    st.markdown("## 📊 Speed Analysis Dashboard")
                    speed_fig = create_speed_analysis_dashboard(st.session_state.df)
                    st.plotly_chart(speed_fig, use_container_width=True)
                    
                elif app_mode == "Delivery Time Prediction":
                    show_prediction_section()
                    
                elif app_mode == "Anomaly Detection":
                    from src.anomaly_detection import detect_anomalies, analyze_anomalies
                    from src.visualization import create_realistic_delivery_plot, create_operational_anomalies_plot
                    
                    # Compute anomalies if not already done
                    if st.session_state.anomalies is None:
                        with st.spinner("Detecting anomalies..."):
                            st.session_state.anomalies = detect_anomalies(st.session_state.X_anomaly_scaled)
                            st.session_state.df_with_anomalies, st.session_state.anomaly_by_city, st.session_state.anomaly_by_type = analyze_anomalies(
                                st.session_state.df_anomaly, st.session_state.anomalies
                            )
                    
                    show_anomaly_section(
                        st.session_state.df_with_anomalies, 
                        st.session_state.anomalies, 
                        st.session_state.anomaly_by_city, 
                        st.session_state.anomaly_by_type
                    )
                    
                    # Operational Anomalies
                    st.markdown("## 🚨 Operational Anomalies")
                    realistic_fig, df_realistic = create_realistic_delivery_plot(st.session_state.df_with_anomalies)
                    st.plotly_chart(realistic_fig, use_container_width=True)
                    
                    # Show operational anomalies table
                    if 'operational_anomaly' in df_realistic.columns:
                        operational_anomalies = df_realistic[df_realistic['operational_anomaly'] != 'normal']
                        if len(operational_anomalies) > 0:
                            st.dataframe(
                                operational_anomalies[
                                    ['order_id', 'from_city_name', 'distance_km', 
                                     'delivery_time_hours', 'speed_kmh_calc', 'operational_anomaly']
                                ].sort_values('delivery_time_hours', ascending=False).head(10),
                                use_container_width=True
                            )
                    else:
                        st.info("No operational anomalies detected in the current dataset.")
                    
                elif app_mode == "Data Insights":
                    show_insights_section(st.session_state.df)
                    
                elif app_mode == "Delivery Map":
                    show_delivery_map(st.session_state.df)
                    
                elif app_mode == "Upload Data":
                    show_data_upload_section()
                    
            except Exception as e:
                st.error(f"❌ Error in application: {str(e)}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())
    else:
        st.error("❌ No data available. Please check your data file.")


if __name__ == "__main__":
    main()

