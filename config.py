import torch
import os

# Global settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
RANDOM_SEED = 42
NUM_EPOCHS = 5
TRAIN_ZIP_PATH = "/content/drive/MyDrive/training_set.zip"
TEST_ZIP_PATH = "/content/drive/MyDrive/test_set.zip"
TRAIN_DIR = "/content/training_data"
TEST_DIR = "/content/test_data"
