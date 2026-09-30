import torch


def pytest_sessionstart(session):
    # Tiny regression fixtures are faster without CPU thread oversubscription.
    torch.set_num_threads(1)
