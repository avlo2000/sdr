from pathlib import Path

import numpy as np
import torch
from torch import utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torchsummary import summary
from ad8302.analysis.despike import Despiker
from ad8302.analysis.load_data import load_data
from ad8302.ml.autoencoders.train_utils import train, vae_train_step
from ad8302.ml.autoencoders.variational_autoencoder import VariationalAutoencoder

BATCH_SIZE = 8
EPOCH_COUNT = 100


def prepare_data(data: np.ndarray):
    return np.sin(np.deg2rad(data)).astype(np.float32).T


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    res = list(load_data(Path('../data/non_labeled_3m_home_31_07.json')))
    all_phs_data = [[], [], [], []]
    for sample in res:
        for i, phs in enumerate(sample.phs_data):
            all_phs_data[i].extend(phs)

    all_phs_data = np.array(all_phs_data)

    plt.subplot(211)
    for data in all_phs_data:
        plt.plot(data)
    data_despiked = all_phs_data.copy()
    data_despiked = Despiker(data_despiked).despike()
    data_prepared = prepare_data(all_phs_data)
    train_data, test_data = train_test_split(data_prepared, test_size=0.2)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = VariationalAutoencoder(torch.tensor(4), 1)
    model = model.to(device)
    train(model, EPOCH_COUNT, train_data_loader, device, vae_train_step)


if __name__ == '__main__':
    main()
