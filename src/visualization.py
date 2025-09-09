# src/visualization.py
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from src.utils import calculate_speed_kmh, is_physically_possible, categorize_delivery_speed, detect_operational_anomalies

def create_delivery_time_histogram(df):
    """
    Create histogram of delivery times
    """
    fig = px.histogram(df, x='delivery_time_hours', nbins=50,
                      title='Distribution of Delivery Times')
    fig.update_layout(xaxis_title='Delivery Time (hours)', yaxis_title='Count')
    return fig

def create_city_performance_chart(df):
    """
    Create chart showing delivery performance by city
    """
    city_stats = df.groupby('from_city_name')['delivery_time_hours'].agg(['mean', 'std']).reset_index()
    fig = px.bar(city_stats, x='from_city_name', y='mean', error_y='std',
                title='Average Delivery Time by City')
    fig.update_layout(xaxis_title='City', yaxis_title='Average Delivery Time (hours)')
    return fig

def create_actual_vs_predicted_plot(y_test, y_pred):
    """
    Create scatter plot of actual vs predicted values
    """
    fig = px.scatter(x=y_test, y=y_pred,
                    labels={'x': 'Actual Delivery Time (hours)', 'y': 'Predicted Delivery Time (hours)'},
                    title='Actual vs Predicted Delivery Times')
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines', name='Perfect Prediction'))
    return fig

def create_anomaly_visualizations(df, anomalies):
    """
    Create visualizations for anomaly detection results
    """
    # Scatter plot of delivery time vs distance with anomalies
    fig = px.scatter(df, x='distance_km', y='delivery_time_hours',
                    color=anomalies, color_continuous_scale=['blue', 'red'],
                    title='Delivery Time vs Distance with Anomalies Highlighted',
                    labels={'distance_km': 'Distance (km)', 'delivery_time_hours': 'Delivery Time (hours)'})
    return fig

def create_delivery_map(df, sample_size=1000):
    """
    Create an interactive map of delivery locations
    """
    # Sample data to avoid overplotting
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create map data - using receipt locations as drop-off points
    map_data = pd.DataFrame({
        'lat': df_sample['receipt_lat'],
        'lon': df_sample['receipt_lng'],
        'delivery_time': df_sample['delivery_time_hours'],
        'city': df_sample['from_city_name'],
        'size': np.sqrt(df_sample['delivery_time_hours']) * 2  # Size by delivery time
    })
    
    return map_data

def create_advanced_delivery_map(df, sample_size=500):
    """
    Create advanced interactive map with Plotly
    """
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    fig = px.scatter_mapbox(
        df_sample,
        lat="receipt_lat",
        lon="receipt_lng",
        color="delivery_time_hours",
        size="delivery_time_hours",
        hover_name="from_city_name",
        hover_data=["delivery_time_hours", "distance_km", "typecode"],
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=15,
        zoom=6,
        height=500,
        title="Delivery Locations Colored by Delivery Time"
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    
    return fig

def create_heatmap(df, sample_size=1000):
    """
    Create a heatmap of delivery density
    """
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create base map centered on Tanzania
    m = folium.Map(location=[-6.3690, 34.8888], zoom_start=6)
    
    # Add heatmap
    heat_data = [[row['receipt_lat'], row['receipt_lng']] for _, row in df_sample.iterrows()]
    HeatMap(heat_data).add_to(m)
    
    return m

def create_city_delivery_map(df):
    """
    Create map showing average delivery time by city with proper coordinate handling
    """
    # Define city coordinates (latitude, longitude) for major Tanzanian cities
    city_coordinates = {
        'Dar es Salaam': (-6.7924, 39.2083),
        'Arusha': (-3.3869, 36.6820),
        'Mwanza': (-2.5164, 32.9176), 
        'Mbeya': (-8.9094, 33.4608),
        'Dodoma': (-6.1630, 35.7516)
    }
    
    # Calculate city statistics
    city_stats = df.groupby('from_city_name').agg({
        'delivery_time_hours': 'mean',
        'order_id': 'count'
    }).reset_index()
    
    # Add coordinates for each city
    city_stats['lat'] = city_stats['from_city_name'].map(lambda x: city_coordinates.get(x, (0, 0))[0])
    city_stats['lon'] = city_stats['from_city_name'].map(lambda x: city_coordinates.get(x, (0, 0))[1])
    
    # Filter out cities without coordinates
    city_stats = city_stats[city_stats['lat'] != 0]
    
    if len(city_stats) == 0:
        return None
    
    # Create the map
    fig = px.scatter_mapbox(
        city_stats,
        lat="lat",
        lon="lon", 
        size="order_id",
        color="delivery_time_hours",
        hover_name="from_city_name",
        hover_data=["delivery_time_hours", "order_id"],
        color_continuous_scale=px.colors.sequential.Plasma,
        size_max=30,
        zoom=5,
        height=500,
        title="Average Delivery Time by City (Size = Number of Deliveries)"
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    
    return fig

def create_realistic_delivery_plot(df, anomalies=None):
    """
    Create delivery time vs distance plot with realistic speed filtering
    and operational anomaly detection
    """
    # Calculate speeds and filter physically possible deliveries
    df = df.copy()
    df['speed_kmh'] = df.apply(lambda x: calculate_speed_kmh(x['distance_km'], x['delivery_time_hours']), axis=1)
    
    # Add physical possibility check
    physical_checks = df.apply(
        lambda x: is_physically_possible(x['distance_km'], x['delivery_time_hours']), 
        axis=1, result_type='expand'
    )
    df[['is_physical', 'speed_calculated', 'speed_category']] = physical_checks
    df.columns = list(df.columns[:-3]) + ['is_physical', 'speed_kmh_calc', 'speed_category']
    
    # Filter only physically possible deliveries
    df_realistic = df[df['is_physical'] == True].copy()
    
    # Detect operational anomalies
    df_realistic['operational_anomaly'] = df_realistic.apply(
        lambda x: detect_operational_anomalies(x['distance_km'], x['delivery_time_hours'], x['speed_kmh_calc']),
        axis=1
    )
    
    # Create the plot
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Delivery Time vs Distance (Realistic Speeds Only)']
    )
    
    # Color map for speed categories
    speed_colors = {
        'critical_slow': 'red',
        'slow': 'orange',
        'normal': 'green',
        'fast': 'blue'
    }
    
    # Add traces for each speed category
    for category in df_realistic['speed_category'].unique():
        category_df = df_realistic[df_realistic['speed_category'] == category]
        fig.add_trace(
            go.Scatter(
                x=category_df['distance_km'],
                y=category_df['delivery_time_hours'],
                mode='markers',
                name=categorize_delivery_speed(20 if category == 'critical_slow' else 40 if category == 'slow' else 120 if category == 'normal' else 121),
                marker=dict(
                    color=speed_colors.get(category, 'gray'),
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='darkgray')
                ),
                hovertemplate=(
                    'Distance: %{x:.1f} km<br>' +
                    'Time: %{y:.1f} hours<br>' +
                    'Speed: %{customdata:.1f} km/h<br>' +
                    'Category: %{text}<extra></extra>'
                ),
                customdata=category_df['speed_kmh_calc'],
                text=[categorize_delivery_speed(s) for s in category_df['speed_kmh_calc']]
            )
        )
    
    # Add reference lines for speed limits
    max_distance = df_realistic['distance_km'].max() * 1.1
    max_time = df_realistic['delivery_time_hours'].max() * 1.1
    
    # Reference speed lines (3, 20, 40, 120, 900 km/h)
    speed_lines = [3, 20, 40, 120, 900]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    labels = ['Walking pace (3 km/h)', 'Critical slow (20 km/h)', 'Slow (40 km/h)', 
              'Normal (120 km/h)', 'Air speed (900 km/h)']
    
    for speed, color, label in zip(speed_lines, colors, labels):
        x_line = np.linspace(0.1, max_distance, 100)
        y_line = x_line / speed
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                name=label,
                hoverinfo='skip'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Delivery Time vs Distance with Realistic Speed Filtering',
        xaxis_title='Distance (km)',
        yaxis_title='Delivery Time (hours)',
        showlegend=True,
        height=600,
        hovermode='closest'
    )
    
    # Add annotations for speed limits
    annotations = []
    for speed, color, label in zip(speed_lines, colors, labels):
        if speed <= 120:  # Only annotate realistic speeds
            x_pos = max_distance * 0.8
            y_pos = x_pos / speed
            annotations.append(dict(
                x=x_pos, y=y_pos,
                xref="x", yref="y",
                text=label,
                showarrow=True,
                arrowhead=2,
                ax=0, ay=-40,
                bgcolor=color,
                opacity=0.8
            ))
    
    fig.update_layout(annotations=annotations)
    
    return fig, df_realistic

def create_operational_anomalies_plot(df_realistic):
    """
    Create a plot focusing on operational anomalies with improved visibility,
    y-axis fixed to 30 hours, and filter for delivery times <= 30 hours
    """
    # Filter only problematic deliveries and strictly enforce delivery times <= 30 hours
    df_problems = df_realistic[
        (df_realistic['operational_anomaly'] != 'normal') & 
        (df_realistic['delivery_time_hours'] <= 30)
    ].copy()
    
    if len(df_problems) == 0:
        # Create empty plot with message
        fig = go.Figure()
        fig.add_annotation(
            text="No operational anomalies detected within 30 hours! 🎉",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title="Operational Anomalies - None Found",
            height=400,
            yaxis_range=[0, 30]  # Enforce y-axis limit in empty plot
        )
        return fig
    
    # Create plot with better scaling and visibility
    fig = px.scatter(
        df_problems,
        x='distance_km',
        y='delivery_time_hours',
        color='operational_anomaly',
        size=np.log(df_problems['delivery_time_hours'] + 1) * 8,  # Better size scaling
        hover_data=['speed_kmh_calc', 'from_city_name'],
        title='🚨 Operational Anomalies - Deliveries Needing Investigation',
        labels={
            'distance_km': 'Distance (km)',
            'delivery_time_hours': 'Delivery Time (hours)',
            'operational_anomaly': 'Problem Type',
            'speed_kmh_calc': 'Speed (km/h)'
        },
        height=600  # Increased height for better visibility
    )
    
    # Update marker styles for enhanced visibility
    fig.update_traces(
        marker=dict(opacity=0.8, line=dict(width=1.5, color='darkgray')),
        selector=dict(mode='markers')
    )
    
    # Set fixed axis ranges for clarity
    x_range = [0, max(10, df_problems['distance_km'].max() * 1.1)]
    y_range = [0, 30]  # Strictly fixed y-axis to 30 hours
    
    fig.update_xaxes(range=x_range, title_text="Distance (km)")
    fig.update_yaxes(range=y_range, title_text="Delivery Time (hours)", tick0=0, dtick=5)  # Clear tick marks
    
    # Add reference lines for meaningful speed thresholds
    speed_thresholds = [
        (20, 'red', 'Critical Slow (20 km/h)'),
        (40, 'orange', 'Slow (40 km/h)'),
        (60, 'yellow', 'Moderate (60 km/h)'),
        (80, 'green', 'Good (80 km/h)'),
    ]
    
    for speed, color, label in speed_thresholds:
        x_line = np.linspace(0.1, x_range[1], 100)
        y_line = x_line / speed
        
        if y_line[-1] <= y_range[1]:
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    line=dict(color=color, width=2, dash='dash'),
                    name=label,
                    hoverinfo='skip',
                    showlegend=True
                )
            )
    
    # Improve layout for better visibility
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            font=dict(size=12)
        ),
        hovermode='closest',
        font=dict(size=14),
        margin=dict(l=50, r=50, t=100, b=50)  # Adjusted margins for clarity
    )
    
    return fig

def create_speed_analysis_dashboard(df):
    """
    Create comprehensive speed analysis dashboard
    """
    # Calculate speeds
    df = df.copy()
    df['speed_kmh'] = df.apply(lambda x: calculate_speed_kmh(x['distance_km'], x['delivery_time_hours']), axis=1)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Speed Distribution',
            'Speed vs Distance',
            'Speed by City',
            'Speed by Service Type'
        ]
    )
    
    # 1. Speed Distribution
    fig.add_trace(
        go.Histogram(x=df['speed_kmh'], nbinsx=50, name='Speed Distribution'),
        row=1, col=1
    )
    
    # 2. Speed vs Distance
    fig.add_trace(
        go.Scatter(x=df['distance_km'], y=df['speed_kmh'], mode='markers', name='Speed vs Distance'),
        row=1, col=2
    )
    
    # 3. Speed by City
    city_speeds = df.groupby('from_city_name')['speed_kmh'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=city_speeds['from_city_name'], y=city_speeds['speed_kmh'], name='Speed by City'),
        row=2, col=1
    )
    
    # 4. Speed by Service Type
    service_speeds = df.groupby('typecode')['speed_kmh'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=service_speeds['typecode'], y=service_speeds['speed_kmh'], name='Speed by Service Type'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Comprehensive Speed Analysis Dashboard"
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_xaxes(title_text="Distance (km)", row=1, col=2)
    fig.update_xaxes(title_text="City", row=2, col=1)
    fig.update_xaxes(title_text="Service Type", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=2)
    fig.update_yaxes(title_text="Average Speed (km/h)", row=2, col=1)
    fig.update_yaxes(title_text="Average Speed (km/h)", row=2, col=2)
    
    return fig