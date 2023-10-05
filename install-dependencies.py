import os

modules = [
  "click~=8.0",
  "python-dotenv",
  "filelock>=3.12.2",
  "torch~=2.0.1",
  "torchvision~=0.15.2",
  "flwr~=1.4.0",
  "zod[cli]==0.3.3",
  "matplotlib~=3.7.1",
  "numpy~=1.23.5",
  "opencv_python~=4.7.0.72",
  "pytorch_lightning~=2.0.7",
]

for module in modules:
    os.system(f"pip install {module}")