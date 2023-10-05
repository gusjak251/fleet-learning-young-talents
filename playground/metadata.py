import sys

sys.path.append('../fleet-learning-young-talents')

from zod import ZodFrames
from zod import constants
from data_partitioner import partition_train_data, PartitionStrategy
from data_loader import load_datasets
import matplotlib.pyplot as plt

NO_CLIENTS = 40

def main() -> None:
    zod_frames = ZodFrames("/mnt/ZOD", version="full")
    partitions = partition_train_data(
        PartitionStrategy.RANDOM,
        NO_CLIENTS,
        zod_frames,
        1.0
    )
    data = load_datasets(partitions, zod_frames)
    print(data)

if __name__ == '__main__':
    main()


