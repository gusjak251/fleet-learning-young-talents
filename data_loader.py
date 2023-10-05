"""Client side data loader."""
from typing import Tuple

from data_partitioner import load_ground_truth
from zod_dataset import ZodDataset
from torch import Generator
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision import transforms
from zod import ZodFrames


def load_datasets(
    partitioned_frame_ids: list, zod_frames: ZodFrames
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaer."""
    seed = 42
    transform = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )

    trainset = ZodDataset(
        zod_frames=zod_frames,
        frames_id_set=partitioned_frame_ids,
        stored_ground_truth=load_ground_truth("/mnt/ZOD/ground_truth.json"),
        transform=transform,
    )

    # Split each partition into train/val and create DataLoader
    len_test = int(len(trainset) * 0.1)
    len_train = int(len(trainset) - len_test)

    lengths = [len_train, len_test]
    ds_train, ds_test = random_split(trainset, lengths, Generator().manual_seed(seed))
    train_sampler = RandomSampler(ds_train)
    trainloader = DataLoader(
        ds_train,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        sampler=train_sampler,
    )
    testloader = DataLoader(ds_test, batch_size=32, num_workers=0)

    return trainloader, testloader