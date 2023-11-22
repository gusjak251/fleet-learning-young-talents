import pandas as pd
from zod import ZodFrames
import json

def load_all_metadata(zod_frames: ZodFrames, sampled_frames: list) -> pd.DataFrame:
    cars = pd.DataFrame()
    for index, i in enumerate(sampled_frames):
        client_frames = sampled_frames[str(index)]
        for subframe in client_frames:
            frame = zod_frames[subframe]
            file = open(frame.info.metadata_path)
            metadata = json.load(file)
            df = pd.DataFrame([metadata])
            cars = pd.concat([cars, df], axis=0)
    return cars

def load_metadata(zod_frames: ZodFrames, sampled_frames: list) -> pd.DataFrame:
    cars = pd.DataFrame()
    for index, sample_frame in enumerate(sampled_frames):
        # client_frames = sampled_frames[str(i)]
        frame = zod_frames[sample_frame]
        metadata = None
        with open(frame.info.metadata_path) as f:
            metadata = json.load(f)
        df = pd.DataFrame([metadata])
        cars = pd.concat([cars, df], axis = 0)
        #     frame = zod_frames[subframe]
        #     file = open(frame.info.metadata_path)
        #     metadata = json.load(file)
        #     df = pd.DataFrame([metadata])
        #     cars = pd.concat([cars, df], axis=0)
    return cars