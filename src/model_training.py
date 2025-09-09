# src/model_training.py
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
import joblib
import streamlit as st
import pandas as pd


def diagnose_data_issues(X, y):
    """Diagnose potential data issues causing poor performance"""
    issues = []
    
    # Check target variable
    if y.std() < 1.0:
        issues.append(f"Low target variance (std={y.std():.3f})")
    
    # Check for constant features
    for i in range(X.shape[1]):
        if np.std(X[:, i]) < 0.001:
            issues.append(f"Constant feature {i}")
    
    # Check for NaN values
    if np.isnan(X).any() or np.isnan(y).any():
        issues.append("NaN values present")
    
    # Check for extreme values
    if (y > 100).any():  # Delivery times > 100 hours
        issues.append("Extreme target values")
    
    return issues

# src/model_training.py - Update train_enhanced_delivery_time_model
def train_enhanced_delivery_time_model(X, y):
    """
    Enhanced model training with better diagnostics and multiple approaches
    """
    # First establish baseline
    baseline_r2 = train_simple_baseline(X, y)
    
    # Diagnose data issues - convert X to numpy array if it's a DataFrame
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X
        
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = y
        
    issues = diagnose_data_issues(X_array, y_array)
    if issues:
        st.warning("⚠️ Data issues found:")
        for issue in issues:
            st.write(f"• {issue}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Try simpler models first
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM' : LGBMRegressor(random_state=42, verbosity=-1),
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)
    }
    
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    results = {}
    
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'test_r2': test_r2,
                    'model': model
                }
                
                st.write(f"**{name}**: Test R² = {test_r2:.3f}")
                
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
    
    if best_model is None:
        st.error("❌ All models failed to train!")
        return None, None, None, None, None, None, None
    
    # Compare with baseline
    improvement = best_r2 - baseline_r2
    st.success(f"✅ Best model: {best_model_name} with R² = {best_r2:.3f}")
    st.info(f"📈 Improvement over baseline: {improvement:.3f}")
    
    # Get final predictions from best model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    joblib.dump(best_model, 'models/delivery_time_model_enhanced.pkl')
    
    return best_model, X_test, y_test, y_pred, mae, rmse, r2

def train_delivery_time_model(X, y, test_size=0.2, random_state=42):
    """
    Train a delivery time prediction model (compatibility version)
    """
    return train_enhanced_delivery_time_model(X, y)

def predict_single_delivery(model, input_features, feature_names=None):
    """
    Predict delivery time for a single package
    """
    if feature_names is not None:
        # Get the feature names the model was trained with
        if hasattr(model, 'feature_names_in_'):
            # Use the model's actual feature names
            model_feature_names = model.feature_names_in_
        else:
            # Use the provided feature names
            model_feature_names = feature_names
        
        # Create feature array in the correct order
        features_ordered = np.zeros(len(model_feature_names))
        
        for i, feature_name in enumerate(model_feature_names):
            if feature_name in input_features:
                features_ordered[i] = input_features[feature_name]
            else:
                # Handle missing features (you might want to set a default value)
                features_ordered[i] = 0  # Or appropriate default
        
        features = features_ordered.reshape(1, -1)
    else:
        # Assume input_features is already in correct format
        features = np.array([list(input_features.values())]).reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    
    return max(0.1, prediction)  # Ensure positive time

def analyze_model_performance(y_true, y_pred, model_name=""):
    """
    Analyze and display detailed model performance metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate error statistics
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Percentage within thresholds
    within_1_hour = np.mean(np.abs(errors) <= 1.0) * 100
    within_2_hours = np.mean(np.abs(errors) <= 2.0) * 100
    within_4_hours = np.mean(np.abs(errors) <= 4.0) * 100
    
    # Display metrics in Streamlit
    st.markdown(f"### 📊 {model_name} Performance Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MAE", f"{mae:.3f} hours")
        st.metric("RMSE", f"{rmse:.3f} hours")
        st.metric("R² Score", f"{r2:.3f}")
    
    with col2:
        st.metric("Mean Error", f"{mean_error:.3f} hours")
        st.metric("Error Std Dev", f"{std_error:.3f} hours")
    
    with col3:
        st.metric("Within 1 hour", f"{within_1_hour:.1f}%")
        st.metric("Within 2 hours", f"{within_2_hours:.1f}%")
        st.metric("Within 4 hours", f"{within_4_hours:.1f}%")
    
    # Interpretation
    if r2 > 0.7:
        st.success("✅ Excellent model performance")
    elif r2 > 0.5:
        st.info("📈 Good model performance")
    elif r2 > 0.3:
        st.warning("⚠️ Moderate model performance - consider improving data quality")
    else:
        st.error("❌ Poor model performance - check data quality and feature engineering")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_error': mean_error,
        'std_error': std_error,
        'within_1_hour': within_1_hour,
        'within_2_hours': within_2_hours,
        'within_4_hours': within_4_hours
    }

def hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning for the model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Perform grid search
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    with st.spinner("Performing hyperparameter tuning (this may take a while)..."):
        grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate best model
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    st.success(f"✅ Best parameters found: {grid_search.best_params_}")
    st.success(f"✅ Best model R² score: {r2:.3f}")
    
    # Save best model
    joblib.dump(best_model, 'models/delivery_time_model_tuned.pkl')
    
    return best_model, grid_search.best_params_


# src/model_training.py - Add this function
def train_simple_baseline(X, y):
    """
    Train a simple baseline model to establish performance benchmark
    """
    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Baseline model: always predict the mean
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    
    # Calculate baseline metrics
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    r2_baseline = r2_score(y_test, y_pred_baseline)
    
    st.write(f"📊 Baseline Model (predict mean):")
    st.write(f"   MAE: {mae_baseline:.3f} hours, RMSE: {rmse_baseline:.3f} hours, R²: {r2_baseline:.3f}")
    
    return r2_baseline

def evaluate_feature_importance(model, feature_names):
    """
    Evaluate and display feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        st.markdown("### 🎯 Feature Importance")
        
        # Display top features
        for i, idx in enumerate(indices[:10]):  # Show top 10 features
            if idx < len(feature_names):
                st.write(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Create feature importance plot
        try:
            import plotly.express as px
            
            # Prepare data for plotting
            top_indices = indices[:min(15, len(feature_names))]
            importance_data = {
                'Feature': [feature_names[i] for i in top_indices],
                'Importance': [importances[i] for i in top_indices]
            }
            
            fig = px.bar(importance_data, x='Importance', y='Feature', 
                         orientation='h', title='Top Feature Importances')
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.info("Install plotly for feature importance visualization: pip install plotly")
    
    else:
        st.warning("Feature importance not available for this model type")

def train_model_with_feature_selection(X, y, threshold=0.01):
    """
    Train model with feature selection based on importance
    """
    # First train a model to get feature importance
    initial_model = RandomForestRegressor(n_estimators=50, random_state=42)
    initial_model.fit(X, y)
    
    # Select features above importance threshold
    if hasattr(initial_model, 'feature_importances_'):
        important_features = initial_model.feature_importances_ > threshold
        X_selected = X[:, important_features]
        
        st.info(f"Selected {np.sum(important_features)} out of {X.shape[1]} features")
        
        # Train final model with selected features
        return train_enhanced_delivery_time_model(X_selected, y)
    else:
        st.warning("Could not perform feature selection, using all features")
        return train_enhanced_delivery_time_model(X, y)