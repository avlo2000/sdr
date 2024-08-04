from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from ad8302.analysis.despike import Despiker
from ad8302.analysis.load_data import load_data
from ad8302.arrays import create_grably_array
from ad8302.beamformer import Beamformer
from ad8302.ml.manifolder import PhaseManifoldLoss, InterferometryPredictor, PhaseProjectionModel, DOAPatternLoss

BATCH_SIZE = 1
EPOCH_COUNT = 50


def transform_dataset(dataset: np.ndarray):
    return np.deg2rad(dataset).astype(np.float32).T


def detransform_dataset(dataset: np.ndarray):
    return np.rad2deg(dataset)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Device: {device}")

    res = list(load_data(Path('../data/dataset3m_home_31_07.json')))
    # res = list(load_data(Path('../data/dataset3m_home_31_07.json')))
    all_phs_data = [[], [], [], []]
    for sample in res:
        for i, phs in enumerate(sample.phs_data):
            all_phs_data[i].extend(phs)

    all_phs_data = np.array(all_phs_data)

    data_despiked = all_phs_data.copy()
    data_despiked = Despiker(data_despiked).despike()
    data_prepared = transform_dataset(data_despiked)
    train_data, test_data = train_test_split(data_prepared, test_size=0.2)
    train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train data size: {len(train_data_loader)}")
    print(f"Test data size: {len(test_data_loader)}")

    freq = 0.433e+9
    ant_locs, topology = create_grably_array(freq)

    interferometer = InterferometryPredictor(freq, ant_locs, topology)
    phase_projection_model = PhaseProjectionModel(len(topology))

    loss_fn = DOAPatternLoss(freq, ant_locs, topology, 1000)
    optimizer = optim.Adam([
            {'params': interferometer.parameters()},
            {'params': phase_projection_model.parameters()}
        ],
        lr=0.001,
        betas=(0.95, 0.99)
    )
    for epoch in range(EPOCH_COUNT):
        print(f'Epoch {epoch + 1}/{EPOCH_COUNT}')
        total_loss = 0.0
        for phases in train_data_loader:
            optimizer.zero_grad()
            phases_projected = phase_projection_model(phases)
            loss = loss_fn(phases_projected)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f"Total train loss: {total_loss}")
        print(f"Avg train loss: {total_loss / len(train_data_loader)}")
        total_loss = 0.0
        for phases in test_data_loader:
            phases_projected = phase_projection_model(phases)
            # doa = interferometer(phases_projected)
            loss = loss_fn(phases_projected)
            total_loss += loss
        print(f"Total test loss: {total_loss}")
        print(f"Avg test loss: {total_loss / len(train_data_loader)}")

    reconstructed_phases = []
    doas = []
    print(phase_projection_model.model.weight)
    print(phase_projection_model.model.bias)
    beamformer = Beamformer(freq, ant_locs, topology)
    for phs in data_prepared:
        phs = phase_projection_model(torch.tensor(phs))
        doa = interferometer(phs)

        reconstructed_phases.append(detransform_dataset(phs.detach().numpy()))
        doas.append(doa.detach().numpy())
        doas_pattern, errors = beamformer.doa_pattern(np.deg2rad(reconstructed_phases[-1]), 1000)
        # plt.plot(doas_pattern, errors)
        # plt.show()

    reconstructed_phases = np.array(reconstructed_phases)
    plt.subplot(311)
    plt.plot(data_despiked.T)
    plt.subplot(312)
    plt.plot(reconstructed_phases)
    plt.subplot(313)
    plt.plot(doas)
    plt.show()


if __name__ == '__main__':
    main()


# tensor([[0.1986, 0.1492, 0.6472, 0.1256],
#         [0.0980, 0.0734, 0.3240, 0.0652],
#         [0.0992, 0.0724, 0.3244, 0.0659],
#         [0.1981, 0.1481, 0.6487, 0.1252]], requires_grad=True)
# Parameter containing:
# tensor([0.2876, 0.1416, 0.1423, 0.2882], requires_grad=True)
