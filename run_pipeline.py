from src.pipeline import MLPipeline

# Create pipeline - read params and data
pipeline = MLPipeline()
pipeline.read_params()
pipeline.read_data()

# EDA reports
#pipeline.create_eda_reports()

# Data Preprocessing
pipeline.data_preprocessing()

# Feature Selection
#pipeline.feature_selection()

# Data Science
#pipeline.create_basic_model()

# Hyperopt
pipeline.hyperopt()

# Prediction
pipeline.make_prediction()

# Validation - dostępne tylko, gdy nie jest stosowany clustering - wykonywany tylko dla modelu ({model_name}_clust0)
#pipeline.validation()