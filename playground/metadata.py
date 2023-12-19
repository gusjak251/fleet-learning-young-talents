import sys

sys.path.append('../')

from zod import ZodFrames
from zod import constants
from data_partitioner import partition_train_data, PartitionStrategy
from data_loader import load_datasets
import matplotlib.pyplot as plt
import json
import pandas as pd
import geopandas as gpd
import os
from datetime import datetime

NO_CLIENTS = 100

def main() -> None:
    zod_frames = ZodFrames("/mnt/ZOD", version="full")
    sampled_frames = partition_train_data(
        PartitionStrategy.RANDOM,
        NO_CLIENTS,
        zod_frames,
        0.2
    )
    cars = pd.DataFrame()
    for index, i in enumerate(sampled_frames):
        client_frames = sampled_frames[str(i)]
        for subframe in client_frames:
            frame = zod_frames[subframe]
            file = open(frame.info.metadata_path)
            metadata = json.load(file)
            if index == 0:
                cars = pd.DataFrame([metadata])
            else:
                df = pd.DataFrame([metadata])
                cars = pd.concat([cars, df], axis=0)
    # metadata = json.load(metadata_path)
    print(cars.head())
    print(cars.columns)

    now = datetime.now().isoformat()
    dir = f"plots/plot_{now}"
    os.system(f"mkdir {dir}")

    plt.figure()
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries.plot(color="lightgrey")
    colors, _ = pd.factorize(cars['road_condition'])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.scatter(cars['longitude'], cars['latitude'], c=colors, s=0.2)
    plt.savefig(f"{dir}/location.png")

    plt.figure()
    ax = cars['country_code'].value_counts(sort=False).plot.bar(rot=30)
    ax.figure.savefig(f"{dir}/countries.png")

    plt.figure()
    ax = cars['time_of_day'].value_counts(sort=False).plot.bar(rot=30)
    ax.figure.savefig(f"{dir}/time.png")

    plt.figure()
    ax = cars['scraped_weather'].value_counts(sort=False).plot.bar(rot=30)
    ax.figure.savefig(f"{dir}/weather.png")

    plt.figure()
    ax = cars['road_condition'].value_counts(sort=False).plot.bar(rot=30)
    ax.figure.savefig(f"{dir}/road_conditions.png")


if __name__ == '__main__':
    main()


