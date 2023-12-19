import time
from data_loader import load_datasets
from metadata_loader import load_all_metadata, save_metadata
from novelty_detection import detect_outliers
from plots import plot_location, plot_accuracy, save_loss_data
from models import Net
from utilities import train, test, get_parameters, set_parameters
import matplotlib as plt
import torch
import random
from zod import ZodFrames
from zod import constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from datetime import datetime
from fleet_aggregators import BaseStrategy
from data_partitioner import PartitionStrategy, partition_train_data


GLOBAL_ROUNDS = 40 #40
NO_CLIENTS = 40 #40
CLIENTS_PER_ROUND = 10 #10
PERCENT_DATA = 0.01
LR = 0.01

PARTITION_STRATEGY = PartitionStrategy.LOCATION


# client selection strategy
def select_client(clients):
    return random.sample(clients,CLIENTS_PER_ROUND)

def main() -> None:

    all_frame_ids = pd.DataFrame(columns=["data"])

    global net
    # GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"{device} is available")

    # Pictures
    zod_frames = ZodFrames("/mnt/ZOD", version="full")
    # takes the parameters from the clients and aggregates them into one set of parameters
    net = Net().to(device)
    # NO_CLIENTS to str
    clients = [str(i) for i in range(NO_CLIENTS)]


    # partitions takes the data and splits it into partitions

    partitions = partition_train_data(
        PARTITION_STRATEGY,
        NO_CLIENTS,
        zod_frames,
        PERCENT_DATA
    )

    now = datetime.now().isoformat()
    savedir = f"sessions/session_{now}"
    os.system(f"mkdir {savedir}")

    metadata = load_all_metadata(zod_frames, partitions)

    metadata = detect_outliers(metadata)

    save_metadata(metadata, f'{savedir}/metadata.csv')

    # Create path for plots
    now = datetime.now().isoformat()
    dir = f"plots/data/plot_{now}"
    os.system(f"mkdir {dir}")

    # Save plots
    plot_location(dir, metadata)

    _, testloader = load_datasets(zod_frames.get_split(constants.VAL), zod_frames)
    round_test_losses = []
    batch_test_losses = []
    batch_train_losses = []
    batch_valid_losses_plot = []

    strategy = BaseStrategy()

    for round in range(1, GLOBAL_ROUNDS+1):
        print("ROUND", round)
        selected = select_client(clients)
        nets = []
        # takes each client and trains them

        for client_idx in selected:
            net_copy = net.to(device)

            net_copy.load_state_dict(net.state_dict())
            net_copy.train()

            trainloader, valloader = load_datasets(partitions[str(client_idx)], zod_frames)

            # net_copy is the model, trainloader is the data, valloader ia the validation data
            epoch_train_losses, epoch_val_losses = train(net_copy, trainloader, valloader, epochs=5)
            for elements in range(len(epoch_val_losses)):
                val_loss = epoch_val_losses[elements][0] if epoch_val_losses else None
                batch_valid_losses_plot.append(val_loss)
                print(f"Val losses: {val_loss}")
            print(f"Client: {client_idx:>2} Train losses: {epoch_train_losses}")
            # print(f"Client: {client_idx:>2} Train losses: {epoch_train_losses}, Val losses: {epoch_val_losses}")

            batch_train_losses.append(sum(epoch_train_losses)/len(epoch_train_losses))
            # batch_valid_losses_plot.append(epoch_val_losses)
            # this takes the parameters of the model and returns them as a list
            nets.append((get_parameters(net_copy), 1))

        # havent we already evaluated the model on the test data?
        # yes, but we want to see how the model improves over time
        agg_weights = strategy.aggregate_fit_fedavg(nets)
        
        set_parameters(net, agg_weights[0])

        net.eval()
        batch_test_losses = []

        # this is the test data
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            batch_test_losses.append(net.loss_fn(pred, target).item())


        round_test_losses.append(sum(batch_test_losses)/len(batch_test_losses))
        print(f"Test loss: {round_test_losses[-1]:.4f}")
        # with global round on x-axis and round test losses y-axis
    save_loss_data(batch_test_losses, batch_train_losses, PARTITION_STRATEGY, f'{savedir}/loss.json')
    plot_accuracy(batch_test_losses, batch_train_losses, round_test_losses, batch_valid_losses_plot)
    # print(batch_train_losses_plot)
    print(batch_valid_losses_plot)
    print()
    print(batch_train_losses)




if __name__ == "__main__":
    main()
