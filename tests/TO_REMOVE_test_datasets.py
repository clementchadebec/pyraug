import pytest
import torch

from pyraug.datasets import BaseDataset

# test Dataset
class Test_Dataset:
    @pytest.fixture
    def data(self):
        return torch.tensor([[10.0, 1], [2, 0.0]])

    @pytest.fixture
    def labels(self):
        return torch.tensor([0, 1])

    def test_dataset(self, data, labels):
        dataset = Dataset(data, labels)
        assert torch.all(dataset[0][0] == data[0])
        assert torch.all(dataset[1][1] == labels[1])
        assert torch.all(dataset.data == data)
        assert torch.all(dataset.labels == labels)