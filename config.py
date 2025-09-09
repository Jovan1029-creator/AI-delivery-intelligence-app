# config.py
# Configuration settings for large dataset handling

# Sampling settings
SAMPLE_SIZE_TRAINING = 50000  # Records to use for model training
SAMPLE_SIZE_ANOMALY = 100000  # Records to use for anomaly detection
SAMPLE_SIZE_VISUALIZATION = 10000  # Records to use for visualizations

# Performance settings
USE_APPROXIMATE_DISTANCES = True  # Use faster distance calculation
OPTIMIZE_MEMORY = True  # Use memory-efficient data types

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2