# lryztal.py

import struct
import zlib
from pathlib import Path

import numpy as np
import torch


MAGIC = b"RZIP"
VERSION = 2


_DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.float32: 2,
    torch.float64: 3,
}

_CODE_TO_DTYPE = {
    1: torch.float16,
    2: torch.float32,
    3: torch.float64,
}


def save_rzip(
    tensor: torch.Tensor,
    path,
    block_size: int = 32,
    zlib_level: int = 9,
):
    """
    Losslessly save a 1D floating-point tensor.

    Pipeline:

        float tensor
            ↓
        reinterpret bits as uint16/uint32/uint64
            ↓
        rearrange blocks by standard deviation
            ↓
        zlib

    No numerical quantization occurs.

    The loaded tensor is bit-for-bit identical to the input.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    if tensor.ndim != 1:
        raise ValueError(
            f"tensor must be 1D, got {tuple(tensor.shape)}"
        )

    if tensor.dtype not in _DTYPE_TO_CODE:
        raise TypeError(
            "Supported dtypes: float16, float32, float64"
        )

    if block_size < 1:
        raise ValueError(
            "block_size must be >= 1"
        )

    if not 0 <= zlib_level <= 9:
        raise ValueError(
            "zlib_level must be between 0 and 9"
        )

    path = Path(path)

    # --------------------------------------------------------
    # Move to CPU and make contiguous.
    # --------------------------------------------------------

    x = tensor.detach().cpu().contiguous()

    dtype = x.dtype
    dtype_code = _DTYPE_TO_CODE[dtype]

    # --------------------------------------------------------
    # Reinterpret the floating-point bits.
    #
    # float16 -> uint16
    # float32 -> uint32
    # float64 -> uint64
    #
    # No bits are changed.
    # --------------------------------------------------------

    if dtype == torch.float16:
        bits = x.view(torch.uint16)

    elif dtype == torch.float32:
        bits = x.view(torch.uint32)

    elif dtype == torch.float64:
        bits = x.view(torch.uint64)

    else:
        raise AssertionError

    words = bits.numpy()

    n = len(words)

    # --------------------------------------------------------
    # Pad into blocks.
    # --------------------------------------------------------

    n_blocks = (
        n + block_size - 1
    ) // block_size

    padded_n = n_blocks * block_size

    if padded_n != n:
        words = np.pad(
            words,
            (0, padded_n - n),
            mode="constant",
        )

    blocks = words.reshape(
        n_blocks,
        block_size,
    )

    # --------------------------------------------------------
    # Calculate block standard deviation.
    #
    # We calculate this on the numerical floating-point values,
    # not the integer bit patterns.
    #
    # This determines ordering only; the actual data remains
    # completely unchanged.
    # --------------------------------------------------------

    values = x.numpy()

    if padded_n != n:
        values = np.pad(
            values,
            (0, padded_n - n),
            mode="constant",
        )

    value_blocks = values.reshape(
        n_blocks,
        block_size,
    )

    std = value_blocks.std(
        axis=1,
        dtype=np.float64,
    )

    permutation = np.argsort(
        std,
        kind="stable",
    ).astype(np.uint32)

    rearranged = blocks[
        permutation
    ].reshape(-1)

    # --------------------------------------------------------
    # Compress.
    # --------------------------------------------------------

    compressed = zlib.compress(
        rearranged.tobytes(),
        level=zlib_level,
    )

    # --------------------------------------------------------
    # Write file.
    #
    # Header:
    #
    # magic          4
    # version        1
    # dtype code     1
    # block size     4
    # number blocks  8
    # number values  8
    # payload size   8
    #
    # permutation
    # payload
    # --------------------------------------------------------

    with open(path, "wb") as f:

        f.write(MAGIC)

        f.write(
            struct.pack(
                "<B",
                VERSION,
            )
        )

        f.write(
            struct.pack(
                "<B",
                dtype_code,
            )
        )

        f.write(
            struct.pack(
                "<I",
                block_size,
            )
        )

        f.write(
            struct.pack(
                "<Q",
                n_blocks,
            )
        )

        f.write(
            struct.pack(
                "<Q",
                n,
            )
        )

        f.write(
            struct.pack(
                "<Q",
                len(compressed),
            )
        )

        # Block permutation
        f.write(
            permutation.tobytes()
        )

        # Compressed payload
        f.write(
            compressed
        )


def load_rzip(
    path,
    device=None,
):
    """
    Load a lossless RZIP tensor.

    Returns a torch.Tensor with exactly the same dtype and
    bit representation as the tensor passed to save_rzip().
    """

    path = Path(path)

    with open(path, "rb") as f:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        magic = f.read(4)

        if magic != MAGIC:
            raise ValueError(
                "Not an RZIP file"
            )

        version = struct.unpack(
            "<B",
            f.read(1),
        )[0]

        if version != VERSION:
            raise ValueError(
                f"Unsupported RZIP version: {version}"
            )

        dtype_code = struct.unpack(
            "<B",
            f.read(1),
        )[0]

        if dtype_code not in _CODE_TO_DTYPE:
            raise ValueError(
                f"Unknown dtype code: {dtype_code}"
            )

        dtype = _CODE_TO_DTYPE[
            dtype_code
        ]

        block_size = struct.unpack(
            "<I",
            f.read(4),
        )[0]

        n_blocks = struct.unpack(
            "<Q",
            f.read(8),
        )[0]

        n = struct.unpack(
            "<Q",
            f.read(8),
        )[0]

        payload_size = struct.unpack(
            "<Q",
            f.read(8),
        )[0]

        # ----------------------------------------------------
        # Permutation
        # ----------------------------------------------------

        permutation = np.frombuffer(
            f.read(
                n_blocks * 4
            ),
            dtype=np.uint32,
        ).copy()

        # ----------------------------------------------------
        # Compressed payload
        # ----------------------------------------------------

        compressed = f.read(
            payload_size
        )

    # --------------------------------------------------------
    # Decompress.
    # --------------------------------------------------------

    raw = zlib.decompress(
        compressed
    )

    # --------------------------------------------------------
    # Determine NumPy word dtype.
    # --------------------------------------------------------

    if dtype == torch.float16:
        word_dtype = np.uint16

    elif dtype == torch.float32:
        word_dtype = np.uint32

    elif dtype == torch.float64:
        word_dtype = np.uint64

    else:
        raise AssertionError

    rearranged = np.frombuffer(
        raw,
        dtype=word_dtype,
    )

    # --------------------------------------------------------
    # Undo permutation.
    # --------------------------------------------------------

    blocks = rearranged.reshape(
        n_blocks,
        block_size,
    )

    restored = np.empty_like(
        blocks
    )

    restored[
        permutation
    ] = blocks

    words = restored.reshape(-1)[
        :n
    ]

    # --------------------------------------------------------
    # Convert raw bits back into torch tensor.
    # --------------------------------------------------------

    bits = torch.from_numpy(
        words.copy()
    )

    if dtype == torch.float16:
        result = bits.view(
            torch.float16
        )

    elif dtype == torch.float32:
        result = bits.view(
            torch.float32
        )

    elif dtype == torch.float64:
        result = bits.view(
            torch.float64
        )

    else:
        raise AssertionError

    if device is not None:
        result = result.to(device)

    return result