import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:

    # Create label encoders for columns with non-numerical values
    road_condition_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    weather_encoder = LabelEncoder()
    time_encoder = LabelEncoder()

    # Convert labels to numerical values
    df['road_condition'] = road_condition_encoder.fit_transform(df['road_condition'])
    df['scraped_weather'] = weather_encoder.fit_transform(df['scraped_weather'])
    df['time_of_day'] = time_encoder.fit_transform(df['time_of_day'])
    df['country_code'] = country_encoder.fit_transform(df['country_code'])

    anomaly_inputs = ['road_condition', 'scraped_weather', 'time_of_day', 'country_code']

    model_IF = IsolationForest(contamination=0.05, n_estimators=50, random_state=42)

    model_IF.fit(df[anomaly_inputs])

    df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
    df['anomaly'] = model_IF.predict(df[anomaly_inputs])

    # Convert numerical values to original labels
    df['road_condition'] = road_condition_encoder.inverse_transform(df['road_condition'])
    df['scraped_weather'] = weather_encoder.inverse_transform(df['scraped_weather'])
    df['time_of_day'] = time_encoder.inverse_transform(df['time_of_day'])
    df['country_code'] = country_encoder.inverse_transform(df['country_code'])
    
    return df