import tempfile
import re
from pathlib import Path

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
    serializer = GeneralModuleSerializer()

    dataset, metadata = serializer.serialize(model)

    assert isinstance(dataset, torch.Tensor)
    assert dataset.ndim == 1
    assert dataset.dtype == torch.float32

    # Every serialized parameter should have metadata.
    assert len(metadata["param_names"]) == len(metadata["param_shapes"])


# ---------------------------------------------------------------------------
# Exact round-trip
# ---------------------------------------------------------------------------

def test_serialize_deserialize_round_trip():
    torch.manual_seed(42)

    original = DummyModel()
    original_state = clone_state_dict(original)

    serializer = GeneralModuleSerializer()

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
# Parameter ordering
# ---------------------------------------------------------------------------

def test_parameter_order_is_preserved():
    torch.manual_seed(123)

    model = DummyModel()
    serializer = GeneralModuleSerializer()

    _, metadata = serializer.serialize(model)

    expected_names = list(model.state_dict().keys())

    assert metadata["param_names"] == expected_names


# ---------------------------------------------------------------------------
# Regex exclusion
# ---------------------------------------------------------------------------

def test_exclude_by_pattern():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer()

    dataset, metadata = serializer.serialize(
        model,
        exclude_patterns=[r"embedding"],
    )

    assert "embedding.weight" not in metadata["param_names"]


# ---------------------------------------------------------------------------
# Exact-name exclusion
# ---------------------------------------------------------------------------

def test_exclude_by_exact_name():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer()

    dataset, metadata = serializer.serialize(
        model,
        exclude_names=["layer1.weight"],
    )

    assert "layer1.weight" not in metadata["param_names"]


# ---------------------------------------------------------------------------
# include_only_patterns
# ---------------------------------------------------------------------------

def test_include_only_patterns():
    torch.manual_seed(42)

    model = DummyModel()
    serializer = GeneralModuleSerializer()

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
    serializer = GeneralModuleSerializer()

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

    serializer = GeneralModuleSerializer()

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
# Metadata only contains param_names and param_shapes
# ---------------------------------------------------------------------------

def test_metadata_only_contains_expected_keys():
    model = DummyModel()
    serializer = GeneralModuleSerializer()

    _, metadata = serializer.serialize(model)

    # Verify metadata only contains the expected keys
    expected_keys = {"param_names", "param_shapes"}
    assert set(metadata.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Different tensor dtypes - updated to handle dtype issues
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
    ],
)
def test_round_trip_with_different_dtypes(dtype):
    torch.manual_seed(42)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(10, dtype=dtype))
            self.bias = nn.Parameter(torch.randn(5, dtype=dtype))

    model = SimpleModel()
    original_state = clone_state_dict(model)

    serializer = GeneralModuleSerializer()

    dataset, metadata = serializer.serialize(model)

    reconstructed = serializer.deserialize(
        dataset,
        metadata,
        SimpleModel,
    )

    assert_state_dict_equal(
        reconstructed.state_dict(),
        original_state,
    )


# ---------------------------------------------------------------------------
# Test with mixed dtypes in the same model
# ---------------------------------------------------------------------------

def test_round_trip_with_mixed_dtypes():
    torch.manual_seed(42)

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.float_param = nn.Parameter(torch.randn(10, dtype=torch.float32))
            self.double_param = nn.Parameter(torch.randn(5, dtype=torch.float64))

    model = MixedModel()
    original_state = clone_state_dict(model)

    serializer = GeneralModuleSerializer()

    dataset, metadata = serializer.serialize(model)

    reconstructed = serializer.deserialize(
        dataset,
        metadata,
        MixedModel,
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
    serializer = GeneralModuleSerializer()

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
    serializer = GeneralModuleSerializer()

    with pytest.raises(re.error):
        serializer.serialize(
            model,
            exclude_patterns=["["],
        )