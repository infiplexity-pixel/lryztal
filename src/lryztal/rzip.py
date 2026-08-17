import struct
import zlib
from pathlib import Path

import numpy as np
import torch


MAGIC = b"RZIP"
VERSION = 1

BLOCK_SIZE = 32
ZLIB_LEVEL = 9


def save_rzip(
    tensor: torch.Tensor,
    path,
    block_size: int = BLOCK_SIZE,
    zlib_level: int = ZLIB_LEVEL
):
    """
    Lossy float32/float64 -> 16-bit RZIP compression.

    The tensor is quantized to int16, blocks are reordered by
    their standard deviation, then compressed with zlib-9.

    The permutation is stored, so the 16-bit representation is
    reconstructed exactly. The floating-point reconstruction is
    approximate because of 16-bit quantization.

    Input:
        1D floating-point torch.Tensor

    Returns:
        None
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    if tensor.ndim != 1:
        raise ValueError(
            f"tensor must be 1D, got shape {tuple(tensor.shape)}"
        )

    if not tensor.is_floating_point():
        raise TypeError(
            f"tensor must be floating point, got {tensor.dtype}"
        )

    path = Path(path)

    # --------------------------------------------------------
    # Move to CPU float32
    # --------------------------------------------------------

    x = (
        tensor.detach()
        .cpu()
        .float()
        .numpy()
    )

    n = len(x)

    # --------------------------------------------------------
    # 16-bit symmetric quantization
    # --------------------------------------------------------

    scale = float(np.max(np.abs(x)))

    if scale == 0.0:
        scale = 1.0

    q = np.rint(
        x / scale * 32767.0
    )

    q = np.clip(
        q,
        -32767,
        32767,
    ).astype(np.int16)

    # Store as uint16 so the byte representation is simple.
    data = (
        q.astype(np.int32) + 32767
    ).astype(np.uint16)

    # --------------------------------------------------------
    # Pad into blocks
    # --------------------------------------------------------

    n_blocks = (
        n + block_size - 1
    ) // block_size

    padded_n = n_blocks * block_size

    if padded_n != n:
        data = np.pad(
            data,
            (0, padded_n - n),
            mode="constant",
        )

    blocks = data.reshape(
        n_blocks,
        block_size,
    )

    # --------------------------------------------------------
    # Reorder blocks by standard deviation
    # --------------------------------------------------------

    signed = (
        blocks.astype(np.int32)
        - 32767
    )

    std = signed.std(
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
    # Compress
    # --------------------------------------------------------

    compressed = zlib.compress(
        rearranged.tobytes(),
        level=zlib_level,
    )

    # --------------------------------------------------------
    # Save
    #
    # Header:
    #
    # magic       4 bytes
    # version     1
    # dtype       1
    # block size  4
    # n blocks    4
    # n values    8
    # scale       8
    # permutation N * 4
    # payload     variable
    # --------------------------------------------------------

    with open(path, "wb") as f:

        f.write(MAGIC)

        f.write(
            struct.pack(
                "<B",
                VERSION,
            )
        )

        # 16-bit quantized representation
        f.write(
            struct.pack(
                "<B",
                16,
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
                "<I",
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
                "<d",
                scale,
            )
        )

        # Permutation
        f.write(
            permutation.tobytes()
        )

        # Compressed payload size
        f.write(
            struct.pack(
                "<Q",
                len(compressed),
            )
        )

        # Payload
        f.write(
            compressed
        )


def load_rzip(
    path,
    device=None,
    dtype=torch.float32,
):
    """
    Load an RZIP file produced by save_rzip().

    Returns:
        1D torch.Tensor

    The returned tensor is approximately equal to the original
    because the stored representation uses 16-bit quantization.
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

        bits = struct.unpack(
            "<B",
            f.read(1),
        )[0]

        if bits != 16:
            raise ValueError(
                f"Unsupported quantization: {bits} bits"
            )

        block_size = struct.unpack(
            "<I",
            f.read(4),
        )[0]

        n_blocks = struct.unpack(
            "<I",
            f.read(4),
        )[0]

        n = struct.unpack(
            "<Q",
            f.read(8),
        )[0]

        scale = struct.unpack(
            "<d",
            f.read(8),
        )[0]

        # ----------------------------------------------------
        # Permutation
        # ----------------------------------------------------

        permutation = np.frombuffer(
            f.read(n_blocks * 4),
            dtype=np.uint32,
        ).copy()

        # ----------------------------------------------------
        # Payload
        # ----------------------------------------------------

        payload_size = struct.unpack(
            "<Q",
            f.read(8),
        )[0]

        compressed = f.read(
            payload_size
        )

    # --------------------------------------------------------
    # Decompress
    # --------------------------------------------------------

    raw = zlib.decompress(
        compressed
    )

    rearranged = np.frombuffer(
        raw,
        dtype=np.uint16,
    )

    # --------------------------------------------------------
    # Undo block permutation
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

    data = restored.reshape(-1)

    # Remove padding
    data = data[:n]

    # --------------------------------------------------------
    # Convert back to signed quantized values
    # --------------------------------------------------------

    q = (
        data.astype(np.int32)
        - 32767
    ).astype(np.int16)

    # --------------------------------------------------------
    # Dequantize
    # --------------------------------------------------------

    x = (
        q.astype(np.float32)
        / 32767.0
        * np.float32(scale)
    )

    result = torch.from_numpy(
        x
    )

    if dtype != torch.float32:
        result = result.to(dtype)

    if device is not None:
        result = result.to(device)

    return result