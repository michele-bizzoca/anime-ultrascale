#!/usr/bin/env python3

import sys

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


TARGET = 0.90
ITERATIONS = 8


def load_luminance(path: str) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def roundtrip(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape

    target_width = max(1, round(width / factor))
    target_height = max(1, round(height / factor))

    pil = Image.fromarray(image)

    reduced = pil.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )

    restored = reduced.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    return np.asarray(restored, dtype=np.uint8)


def similarity(image: np.ndarray, factor: float) -> float:
    reconstructed = roundtrip(image, factor)

    return float(
        structural_similarity(
            image,
            reconstructed,
            data_range=255,
        )
    )


def detect_factor(image: np.ndarray) -> float:
    low = 1.0
    high = 2.0

    # Find an upper bound whose SSIM is below the target.
    while similarity(image, high) >= TARGET:
        low = high
        high *= 2.0

        height, width = image.shape

        # Nothing meaningful remains beyond 1x1.
        if round(width / high) <= 1 or round(height / high) <= 1:
            break

    # Binary search for SSIM ~= TARGET.
    for _ in range(ITERATIONS):
        mid = (low + high) / 2.0

        if similarity(image, mid) >= TARGET:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} IMAGE")

    image = load_luminance(sys.argv[1])
    factor = detect_factor(image)

    print(factor)


if __name__ == "__main__":
    main()
