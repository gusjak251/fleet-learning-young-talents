import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def plot_location(dir: str, metadata: pd.DataFrame):
    plt.figure()
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries.plot(color="lightgrey")
    colors, _ = pd.factorize(metadata['road_condition'])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.scatter(metadata['longitude'], metadata['latitude'], c=colors, s=0.2)
    plt.savefig(f"{dir}/location.png")