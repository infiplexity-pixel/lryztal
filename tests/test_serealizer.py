import tempfile
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from lryztal import GeneralModuleSerializer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(7, 5)
        self.layer1 = nn.Linear(5, 3)
        self.layer2 = nn.Linear(3, 4)

        # Deliberately different shapes to test arbitrary tensors.
        self.extra = nn.Parameter(torch.randn(2, 3, 4))

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer1(x)
        return self.layer2(x)


def clone_state_dict(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def assert_state_dict_equal(actual, expected, names=None):
    if names is None:
        names = expected.keys()

    for name in names:
        assert name in actual
        assert torch.equal(actual[name], expected[name]), (
            f"Mismatch in parameter: {name}"
        )


# ---------------------------------------------------------------------------
# Basic serialization
# ---------------------------------------------------------------------------

def test_serialize_shape_and_metadata():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(model)

    assert isinstance(dataset, np.ndarray)
    assert dataset.ndim == 2
    assert dataset.shape[1] == 8

    assert metadata["chunk_size"] == 8
    assert metadata["num_rows"] == dataset.shape[0]
    assert metadata["total_params"] > 0

    # Every serialized parameter should have metadata.
    assert len(metadata["param_names"]) == len(metadata["param_shapes"])


# ---------------------------------------------------------------------------
# Exact round-trip
# ---------------------------------------------------------------------------

def test_serialize_deserialize_round_trip():
    torch.manual_seed(42)

    original = DummyModel()
    original_state = clone_state_dict(original)

    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(original)
    reconstructed = serializer.deserialize(
        dataset,
        metadata,
        DummyModel,
    )

    assert_state_dict_equal(
        reconstructed.state_dict(),
        original_state,
    )


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def test_padding_is_correctly_removed():
    torch.manual_seed(42)

    model = DummyModel()

    # Almost certainly forces padding.
    serializer = GeneralModuleSerializer(chunk_size=13)

    dataset, metadata = serializer.serialize(model)

    assert metadata["padding_needed"] >= 0
    assert dataset.shape[1] == 13

    reconstructed = serializer.deserialize(
        dataset,
        metadata,
        DummyModel,
    )

    assert_state_dict_equal(
        reconstructed.state_dict(),
        model.state_dict(),
    )


def test_no_padding_when_exactly_divisible():
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(16).float())

    model = SmallModel()

    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(model)

    assert metadata["total_params"] == 16
    assert metadata["padding_needed"] == 0
    assert dataset.shape == (2, 8)


# ---------------------------------------------------------------------------
# Parameter ordering
# ---------------------------------------------------------------------------

def test_parameter_order_is_preserved():
    torch.manual_seed(123)

    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    _, metadata = serializer.serialize(model)

    expected_names = list(model.state_dict().keys())

    assert metadata["param_names"] == expected_names


# ---------------------------------------------------------------------------
# Regex exclusion
# ---------------------------------------------------------------------------

def test_exclude_by_pattern():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(
        model,
        exclude_patterns=[r"embedding"],
    )

    assert "embedding.weight" not in metadata["param_names"]

    excluded_names = {
        item["name"]
        for item in metadata["excluded_params"]
    }

    assert "embedding.weight" in excluded_names


# ---------------------------------------------------------------------------
# Exact-name exclusion
# ---------------------------------------------------------------------------

def test_exclude_by_exact_name():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(
        model,
        exclude_names=["layer1.weight"],
    )

    assert "layer1.weight" not in metadata["param_names"]

    excluded_names = {
        item["name"]
        for item in metadata["excluded_params"]
    }

    assert "layer1.weight" in excluded_names


# ---------------------------------------------------------------------------
# include_only_patterns
# ---------------------------------------------------------------------------

def test_include_only_patterns():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(
        model,
        include_only_patterns=[r"layer1"],
    )

    assert metadata["param_names"] == [
        "layer1.weight",
        "layer1.bias",
    ]


# ---------------------------------------------------------------------------
# include_only + exclusion
# ---------------------------------------------------------------------------

def test_include_only_pattern_with_exclusion():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(
        model,
        include_only_patterns=[r"layer1"],
        exclude_names=["layer1.bias"],
    )

    assert metadata["param_names"] == [
        "layer1.weight",
    ]


# ---------------------------------------------------------------------------
# Excluded parameters are preserved during deserialization
# ---------------------------------------------------------------------------

def test_excluded_parameters_are_not_overwritten():
    torch.manual_seed(42)

    original = DummyModel()

    serializer = GeneralModuleSerializer(chunk_size=8)

    dataset, metadata = serializer.serialize(
        original,
        exclude_names=["embedding.weight"],
    )

    torch.manual_seed(999)

    reconstructed = serializer.deserialize(
        dataset,
        metadata,
        DummyModel,
    )

    # Serialized parameters should match the original.
    assert torch.equal(
        reconstructed.layer1.weight,
        original.layer1.weight,
    ), "A"

    assert torch.equal(
        reconstructed.layer1.bias,
        original.layer1.bias,
    ), "B"

    assert not torch.equal(
        reconstructed.embedding.weight,
        original.embedding.weight,
    ), "D"


# ---------------------------------------------------------------------------
# Exclusion metadata
# ---------------------------------------------------------------------------

def test_get_excluded_params_info():
    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    _, metadata = serializer.serialize(
        model,
        exclude_patterns=[r"embedding"],
        exclude_names=["layer1.bias"],
    )

    excluded = serializer.get_excluded_params_info(metadata)

    excluded_names = {
        item["name"]
        for item in excluded
    }

    assert "embedding.weight" in excluded_names
    assert "layer1.bias" in excluded_names


# ---------------------------------------------------------------------------
# File round-trip
# ---------------------------------------------------------------------------

def test_serialize_to_files_and_deserialize_from_files():
    torch.manual_seed(42)

    original = DummyModel()
    original_state = clone_state_dict(original)

    serializer = GeneralModuleSerializer(chunk_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        dataset_path = tmp / "dataset.npy"
        metadata_path = tmp / "metadata.pt"

        serializer.serialize_to_files(
            original,
            dataset_path,
            metadata_path,
        )

        assert dataset_path.exists()
        assert metadata_path.exists()

        reconstructed = serializer.deserialize_from_files(
            dataset_path,
            metadata_path,
            DummyModel,
        )

    assert_state_dict_equal(
        reconstructed.state_dict(),
        original_state,
    )


# ---------------------------------------------------------------------------
# Different chunk sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "chunk_size",
    [
        1,
        2,
        3,
        7,
        8,
        16,
        31,
        64,
        128,
    ],
)
def test_round_trip_with_different_chunk_sizes(chunk_size):
    torch.manual_seed(42)

    model = DummyModel()
    original_state = clone_state_dict(model)

    serializer = GeneralModuleSerializer(
        chunk_size=chunk_size,
    )

    dataset, metadata = serializer.serialize(model)

    reconstructed = serializer.deserialize(
        dataset,
        metadata,
        DummyModel,
    )

    assert_state_dict_equal(
        reconstructed.state_dict(),
        original_state,
    )


# ---------------------------------------------------------------------------
# Empty selection
# ---------------------------------------------------------------------------

def test_empty_selection_raises():
    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    with pytest.raises(ValueError, match="No parameters"):
        serializer.serialize(
            model,
            include_only_patterns=[r"THIS_DOES_NOT_EXIST"],
        )


# ---------------------------------------------------------------------------
# Invalid regex behavior is delegated to re.search naturally.
# ---------------------------------------------------------------------------

def test_invalid_pattern_raises():
    model = DummyModel()
    serializer = GeneralModuleSerializer(chunk_size=8)

    with pytest.raises(re.error):
        serializer.serialize(
            model,
            exclude_patterns=["["],
        )