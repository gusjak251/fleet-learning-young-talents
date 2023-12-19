from scipy.stats import shapiro, anderson
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import OrderedDict
import numpy as np
import sys
import seaborn as sns

sys.path.append('../')

import matplotlib.pyplot as plt
from data_partitioner import partition_train_data, PartitionStrategy
from metadata_loader import load_all_metadata
from zod import ZodFrames

NO_CLIENTS = 100

def outlier_plot(data, outlier_method_name, x_var, y_var,
                 xaxis_limits=[0, 1], yaxis_limits=[0, 1]):
    print(xaxis_limits)
    print(yaxis_limits)
    print(f'Outlier Method: {outlier_method_name}')

    # Create a dynamic title based on the method
    method = f'{outlier_method_name}_anomaly'

    # Print out key statistics
    print(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
    print(f"Number of non anomalous values  {len(data[data['anomaly']== 1])}")
    print(f'Total Number of Values: {len(data)}')

    # Create the chart using seaborn
    g = sns.FacetGrid(data, col='anomaly', height=4,
                      hue='anomaly', hue_order=[1, -1])
    g.map(sns.scatterplot, x_var, y_var)
    g.fig.suptitle(
        f'Outlier Method: {outlier_method_name}', y=1.10, fontweight='bold')
    # g.set(xlim=xaxis_limits, ylim=yaxis_limits)
    axes = g.axes.flatten()
    axes[0].set_title(f"Outliers\n{len(data[data['anomaly']== -1])} points")
    axes[1].set_title(f"Inliers\n {len(data[data['anomaly']==  1])} points")
    # Save the plot as PNG
    g.savefig("test.png")
    return g

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

def main() -> None:
    zod_frames = ZodFrames("/mnt/ZOD", version="full")

    partitions = partition_train_data(
        PartitionStrategy.RANDOM,
        NO_CLIENTS,
        zod_frames,
        0.2,
    )
    df = load_all_metadata(zod_frames, partitions)

    df = df.dropna()

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

    outlier_plot(df, 'Isolation Forest', 'longitude',
                'latitude')


if __name__ == '__main__':
    main()
