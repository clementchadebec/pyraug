import numpy as np
import pytest
import torch

from pyraug.data_loader import DataChecker, DataGetter, Dataset
from pyraug.exception.customexception import LoadError

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


# data loading
class Test_Data_Loading:
    @pytest.fixture
    def demo_data_path(self):
        return "src/pyraug/demo/data/mnist_demo_no_targets"

    @pytest.fixture
    def corrupted_path(self):
        return "corrupted_path_to_data"

    def test_load_demo_data(self, demo_data_path):
        data_getter = DataGetter()
        data = data_getter.get_data(demo_data_path)
        assert data is not None

    def test_raise_load_error(self, corrupted_path):
        data_getter = DataGetter()
        with pytest.raises(LoadError):
            data = data_getter.get_data(corrupted_path)


# data format
class Test_Format_Data:
    @pytest.fixture(params=[[[0.0, 1.0]], {"a": 0}])
    def bad_data(self, request):
        return request.param

    @pytest.fixture
    def bad_shape_data(self):
        return np.array([[10.0, 1], [2, 5.0, 0.0], [0.0, 1.0]])

    @pytest.fixture(
        params=[torch.tensor([[np.nan, 0], [12, 0]]), np.array([[np.nan, 0], [12, 0]])]
    )
    def nan_data(self, request):
        return request.param

    @pytest.fixture(
        params=[
            np.array([[10.0, 1], [2, 0.0], [0.0, 1.0]]),
            torch.tensor([[10.0, 1], [2, 0.0], [0.0, 1.0]]),
            torch.tensor([[[10.0, 1], [0.0, 1]], [[2, 0.0], [0.0, 1.0]]]),
        ]
    )
    def unormalized_data(self, request):
        return request.param

    def test_raise_bad_data_error(self, bad_data):
        data_check = DataChecker()
        with pytest.raises(TypeError):
            data_check.check_data(bad_data)

    @pytest.mark.filterwarnings("ignore")
    def test_raise_bad_shape_error(self, bad_shape_data):
        data_check = DataChecker()
        with pytest.raises(TypeError):
            data_check.check_data(bad_shape_data)

    def test_raise_nan_data_error(self, nan_data):
        data_check = DataChecker()
        with pytest.raises(ValueError):
            data_check.check_data(nan_data)

    def test_normalize_data(self, unormalized_data):
        data_check = DataChecker()
        checked_data = data_check.check_data(unormalized_data)
        assert all(checked_data.min(dim=1)[0] <= 0) and all(
            checked_data.max(dim=1)[0] <= 1
        ), f"{checked_data}"

    def test_data_shape(self, unormalized_data):
        data_check = DataChecker()
        checked_data = data_check.check_data(unormalized_data)

        assert (
            checked_data.shape
            == unormalized_data.reshape(unormalized_data.shape[0], -1).shape
        )
