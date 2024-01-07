import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import json
from data_partitioner import PartitionStrategy


# Plot data points on a world map & save the figure as an image
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


# Save loss data in json format
def save_loss_data(batch_test_losses, batch_train_losses, strategy, path: str):
    strat = 'default'
    if strategy == PartitionStrategy.RANDOM:
        strat = 'random'
    if strategy == PartitionStrategy.LOCATION:
        strat = 'location'
    if strategy == PartitionStrategy.ROAD_CONDITION:
        strat = 'road_condition'
    output = {
        'test_loss': batch_test_losses,
        'train_loss': batch_train_losses,
        'strategy': strat
    }
    with open(path, 'w') as outfile:
        json.dump(output, outfile, indent=4)

# Plot loss data & save the figure to an image file
def plot_accuracy(batch_test_losses, batch_train_losses, round_test_losses, batch_valid_losses_plot):
    train_loss=[]
    valid_loss=[]

    plt.figure()
    plt.plot(batch_train_losses, c='orange', label='Train losses')
    plt.plot(batch_test_losses, c='blue', label='test losses')
    plt.legend()


    valid_values=[]
    added_values=0
    counter=0
    for epoch in batch_valid_losses_plot:
        if counter==5:
            mean=(added_values/5)
            valid_values.append(mean)
            counter=0
            added_values=0
        added_values +=epoch
        counter +=1
    plt.plot(valid_values,c="red",label="Valid losses")
    plt.legend()
    
    # plt.legend("Test losses","Train losses",loc="best")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Model loss")


    now = datetime.now().isoformat()
    dir = f"plots/performance/performance_{now}"

    os.system(f"mkdir {dir}")
    plt.savefig(f"{dir}/Test_loss.png")
