import numpy as np
import pytest
import torch
import os

from pyraug.customexception import LoadError

from pyraug.data.loader import DataGetter
from pyraug.data.datasets import BaseDataset

from nibabel.testing import data_path


PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Dataset:
    @pytest.fixture
    def data(self):
        return torch.tensor([[10.0, 1], [2, 0.0]])

    @pytest.fixture
    def labels(self):
        return torch.tensor([0, 1])

    def test_dataset(self, data, labels):
        dataset = BaseDataset(data, labels)
        assert torch.all(dataset[0]['data'] == data[0])
        assert torch.all(dataset[1]['labels'] == labels[1])
        assert torch.all(dataset.data == data)
        assert torch.all(dataset.labels == labels)

class Test_Data_Loading:
    @pytest.fixture(params=[
        os.path.join(PATH, 'data/loading/dummy_data_folder/example0.bmp'),
        os.path.join(PATH, 'data/loading/dummy_data_folder/example0.eps'),
        os.path.join(PATH, 'data/loading/dummy_data_folder/example0.gif'),
        os.path.join(PATH, 'data/loading/dummy_data_folder/example0.jpeg'),
        os.path.join(PATH, 'data/loading/dummy_data_folder/example0.jpg'),
        os.path.join(PATH, 'data/loading/dummy_data_folder/example0.png'),
        os.path.join(data_path, 'example4d.nii.gz')
    ])
    def demo_data(self, request):
        return request.param

    def test_load_in_array(self, demo_data):
        data = DataGetter.load_image(demo_data)
        assert type(data) == np.ndarray

class Test_Data_Loading_From_Folder:
    @pytest.fixture
    def path_to_dummy_data_folder(self):
        return os.path.join(PATH, "data/loading/dummy_data_folder")

    def test_returns_good_type(self, path_to_dummy_data_folder):
        data = DataGetter.load_from_folder(path_to_dummy_data_folder)
        assert type(data) == list

# # data loading
# class Test_Data_Loading:
#     @pytest.fixture
#     def demo_data_path(self):
#         return "src/pyraug/demo/data/mnist_demo_no_targets"
# 
#     @pytest.fixture
#     def corrupted_path(self):
#         return "corrupted_path_to_data"
# 
#     def test_load_demo_data(self, demo_data_path):
#         data_getter = DataGetter()
#         data = data_getter.get_data(demo_data_path)
#         assert data is not None
# 
#     def test_raise_load_error(self, corrupted_path):
#         data_getter = DataGetter()
#         with pytest.raises(LoadError):
#             data = data_getter.get_data(corrupted_path)

