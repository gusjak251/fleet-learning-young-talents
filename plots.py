import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import json
from data_partitioner import PartitionStrategy


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


def save_loss_data(batch_test_losses, batch_train_losses, strategy, path: str):
    # train_values = []
    # for epoch in batch_train_losses:
    #     mean = 0
    #     sum_values = 0
    #     for round in epoch:
    #         sum_values += round
    #     mean = sum_values/len(epoch)
    #     train_values.append(mean)
    # test_values = []
    # for epoch in batch_test_losses:
    #     for round in epoch:
    #         test_values.append(round)
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

# x - rounds , y - accuracy, round_test_losses, train_losses
# array i en array [[1,2,3],[4,5,6]]
# Se till att försöka få tag på endast ett värde i arrayen, omvandla den till int, suma och dela med längden
def plot_accuracy(batch_test_losses, batch_train_losses, round_test_losses, batch_valid_losses_plot):
    print(batch_test_losses)
    print()
    print(batch_train_losses)
    train_loss=[]
    valid_loss=[]
    plt.figure()
    # plt.plot(round_test_losses)

    # train_values = []
    # for epoch in batch_train_losses:
    #     mean = 0
    #     sum_values = 0
    #     for round in epoch:
    #         sum_values += round
    #     mean = sum_values/len(epoch)
    #     train_values.append(mean)
    plt.plot(batch_train_losses, c='orange', label='Train losses')

    # test_values = []
    # for epoch in batch_test_losses:
    #     for round in epoch:
    #         test_values.append(round)
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


    # for i in range(len(batch_test_losses)):
    #     plt.plot(batch_test_losses[i])

    # for i in range(len(batch_train_losses_plot)):
    #      for k in range(len(batch_train_losses_plot[i])):
    #              a=(batch_train_losses_plot[i][k][1])
    #              train_loss.append(a)

    # for i in range(len(batch_valid_losses_plot)):
    #      for k in range(len(batch_valid_losses_plot[i])):
    #              a=(batch_valid_losses_plot[i][k][1])
    #              valid_loss.append(a)

    # plt.plot(train_loss)
    # plt.plot(valid_loss)
    
    # plt.legend("Test losses","Train losses",loc="best")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Model loss")


    now = datetime.now().isoformat()
    dir = f"plots/performance/performance_{now}"

    os.system(f"mkdir {dir}")
    plt.savefig(f"{dir}/Test_loss.png")
