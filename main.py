from data_loader import load_datasets
from data_partitioner import partition_train_data, PartitionStrategy
from fleet_aggregators import aggregate
from models import Net
from utilities import train, test, get_parameters, set_parameters
from zod_dataset import ZodDataset
import matplotlib as plt
import torch
import random
from zod import ZodFrames
from zod import constants

GLOBAL_ROUNDS = 40
NO_CLIENTS = 40
CLIENTS_PER_ROUND = 10
LR = 0.001

# client selection strategy
def select_client(clients):
    return random.sample(clients,CLIENTS_PER_ROUND)

def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    zod_frames = ZodFrames("/mnt/ZOD", version="full")

    net = Net().to(device)
    clients = [str(i) for i in range(NO_CLIENTS)]

    partitions = partition_train_data(
        PartitionStrategy.RANDOM,
        NO_CLIENTS,
        zod_frames,
        1.0
    )
    testloader = load_datasets(zod_frames.get_split(constants.VAL), zod_frames)

    round_test_losses = []

    for round in range(1, GLOBAL_ROUNDS+1):
        print("ROUND", round)
        selected = select_client(clients)
        nets = []
        for client_idx in selected:
            net_copy = Net().to(device)
            net_copy.load_state_dict(net.state_dict())
            net_copy.train()
            trainloader, valloader = load_datasets(partitions[str(client_idx)], zod_frames)

            epoch_train_losses, epoch_val_losses = train(net_copy, trainloader, valloader, epochs=5)
            print(f"Client: {client_idx:>2} Train losses: {epoch_train_losses}, Val losses: {epoch_val_losses}")
            
            nets.append((get_parameters(net_copy), 1))
        
        set_parameters(net, aggregate(nets))

        net.eval()
        batch_test_losses = []
        for data,target in testloader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            batch_test_losses.append(net.loss_fn(pred, target).item())
        round_test_losses.append(sum(batch_test_losses)/len(batch_test_losses))
        print(f"Test loss: {round_test_losses[-1]:.4f}")

        # clean up and maybe debug on the code above might be needed.
        # add some plot of the results after the loop, like a graph
        # with global round on x-axis and round test losses y-axis


if __name__ == "__main__":
    main()