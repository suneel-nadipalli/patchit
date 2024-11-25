import os
import pytest
from patchviz import DatasetLoader
from torchvision.datasets import ImageFolder

@pytest.fixture
def setup_loader(tmp_path):
    # Create temporary directories and files for a mock dataset
    class_names = ["class1", "class2"]
    for class_name in class_names:
        os.makedirs(tmp_path / class_name)
        with open(tmp_path / class_name / "image.jpg", "w") as f:
            f.write("dummy data")

    loader = DatasetLoader(data_dir=tmp_path, batch_size=2)
    return loader, class_names

def test_data_loader_initialization(setup_loader):
    loader, class_names = setup_loader

    assert isinstance(loader.dataset, ImageFolder), "Dataset is not an ImageFolder instance."
    assert set(loader.class_to_idx.keys()) == set(class_names), "Class names don't match."

def test_sample_batch(setup_loader):
    loader, _ = setup_loader

    images, labels = loader.get_sample_batch()
    assert len(images) > 0, "No images returned in the batch."
    assert len(labels) > 0, "No labels returned in the batch."

def test_plot_sample_batch(setup_loader):
    loader, _ = setup_loader

    try:
        loader.plot_sample_batch()
    except Exception as e:
        pytest.fail(f"Plot sample batch raised an exception: {e}")
