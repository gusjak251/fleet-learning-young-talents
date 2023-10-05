"""Partition data and distribute to clients."""
import random
from enum import Enum

from zod import constants
from zod import ZodFrames
import numpy as np
import json

random.seed(2023)

class PartitionStrategy(Enum):
    """Partition Strategy enum."""

    RANDOM = "random"
    LOCATION = "location"

# load data based on cid and strategy
def partition_train_data(
    strat: PartitionStrategy,
    no_clients: int,
    zod_frames: ZodFrames,
    percentage_of_data: int,
) -> dict:
    """Partition train data.

    Data partition will be saved as a dictionary client_number -> [frames_id's] and this
    dict is downloaded by the client that loads the correct elements by the idx list
    in the dictionary.

    Args:
        strat (PartitionStrategy): partition strategy
        no_clients (int): number of clients
        zod_importer (ZODImporter): dataset importer
        percentage_of_data (int): percentage of data to partition

    Returns:
        dict: client_number -> frames_id map
    """
    training_frames_all = zod_frames.get_split(constants.TRAIN)

    ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")
    print("loaded stored ground truth")

    training_frames_all = [
        idx for idx in training_frames_all if is_valid_frame(idx, ground_truth)
    ]

    # randomly sample by percentage of data
    sampled_training_frames = random.sample(
        training_frames_all, int(len(training_frames_all) * percentage_of_data)
    )

    if strat == PartitionStrategy.RANDOM:
        cid_partitions = {}
        random.shuffle(sampled_training_frames)
        sublist_size = len(sampled_training_frames) // no_clients
        for i in range(no_clients):
            cid_partitions[str(i)] = sampled_training_frames[
                i * sublist_size : (i + 1) * sublist_size
            ]

    if strat == PartitionStrategy.LOCATION:
        pass

    return cid_partitions


def is_valid_frame(frame_id: str, ground_truth: dict) -> bool:
    """Check if frame is valid."""
    if frame_id == "005350":
        return False

    return frame_id in ground_truth


def load_ground_truth(path: str) -> dict:
    """Load ground truth from file."""
    with open(path) as json_file:
        gt = json.load(json_file)

    for f in gt:
        gt[f] = np.array(gt[f])

    return gt