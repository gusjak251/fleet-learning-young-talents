import sys

sys.path.append('../fleet-learning-young-talents')

from zod import ZodFrames
from zod import constants
from data_partitioner import partition_train_data, PartitionStrategy
from data_loader import load_datasets
import matplotlib.pyplot as plt
import json

NO_CLIENTS = 40

def main() -> None:
    zod_frames = ZodFrames("/mnt/ZOD", version="full")
    for i in range(20):
        frame = zod_frames[str(i)]
        file = open(frame.info.metadata_path)
        metadata = json.load(file)
        print(metadata)
    # metadata = json.load(metadata_path)
    exit()
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


