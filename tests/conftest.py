"""Small deterministic CPU models do not benefit from large thread pools."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(2)
