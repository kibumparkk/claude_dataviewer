"""
Sampling algorithms for large dataset visualization.
Implements LTTB (Largest Triangle Three Buckets) and simple sampling.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def _to_numeric(arr: np.ndarray) -> np.ndarray:
    """
    Convert array to numeric type for calculations.
    Handles datetime types by converting to nanoseconds.

    Args:
        arr: Input array

    Returns:
        Numeric array suitable for calculations
    """
    if np.issubdtype(arr.dtype, np.datetime64):
        # Convert datetime to numeric (nanoseconds since epoch)
        return arr.astype('datetime64[ns]').astype(np.float64)
    elif np.issubdtype(arr.dtype, np.timedelta64):
        # Convert timedelta to numeric (nanoseconds)
        return arr.astype('timedelta64[ns]').astype(np.float64)
    elif not np.issubdtype(arr.dtype, np.number):
        # Try to convert to float, fall back to index if fails
        try:
            return arr.astype(np.float64)
        except (ValueError, TypeError):
            return np.arange(len(arr), dtype=np.float64)
    return arr.astype(np.float64)


def lttb_downsample(x: np.ndarray, y: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Largest Triangle Three Buckets (LTTB) downsampling algorithm.

    This algorithm selects points that best preserve the visual appearance
    of the data when plotted.

    Args:
        x: X-axis data array
        y: Y-axis data array
        threshold: Target number of points after downsampling

    Returns:
        Tuple of downsampled (x, y) arrays
    """
    n = len(x)

    if threshold >= n or threshold < 3:
        return x, y

    # Convert to numeric for calculations
    x_num = _to_numeric(x)
    y_num = _to_numeric(y)

    # Always keep first and last points
    sampled_indices = [0]

    # Bucket size (minus first and last points)
    bucket_size = (n - 2) / (threshold - 2)

    a = 0  # Index of the point in the previous bucket

    for i in range(threshold - 2):
        # Calculate bucket range
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)

        # Calculate average point of next bucket (for triangle area calculation)
        next_bucket_start = bucket_end
        next_bucket_end = int((i + 3) * bucket_size) + 1
        next_bucket_end = min(next_bucket_end, n)

        if next_bucket_start < next_bucket_end:
            avg_x = np.mean(x_num[next_bucket_start:next_bucket_end])
            avg_y = np.mean(y_num[next_bucket_start:next_bucket_end])
        else:
            avg_x = x_num[-1]
            avg_y = y_num[-1]

        # Find point in current bucket with maximum triangle area
        max_area = -1.0
        max_idx = bucket_start

        point_a_x = x_num[a]
        point_a_y = y_num[a]

        for j in range(bucket_start, bucket_end):
            # Calculate triangle area using cross product
            area = abs(
                (point_a_x - avg_x) * (y_num[j] - point_a_y) -
                (point_a_x - x_num[j]) * (avg_y - point_a_y)
            )

            if area > max_area:
                max_area = area
                max_idx = j

        sampled_indices.append(max_idx)
        a = max_idx

    # Always include the last point
    sampled_indices.append(n - 1)

    return x[sampled_indices], y[sampled_indices]


def simple_downsample(x: np.ndarray, y: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple uniform interval downsampling.

    Selects evenly spaced points from the data.

    Args:
        x: X-axis data array
        y: Y-axis data array
        threshold: Target number of points after downsampling

    Returns:
        Tuple of downsampled (x, y) arrays
    """
    n = len(x)

    if threshold >= n or threshold < 2:
        return x, y

    indices = np.linspace(0, n - 1, threshold, dtype=int)
    return x[indices], y[indices]


def apply_sampling(x: np.ndarray, y: np.ndarray, mode: str, sample_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply sampling based on the selected mode.

    Args:
        x: X-axis data array
        y: Y-axis data array
        mode: Sampling mode ('full', 'lttb', 'simple')
        sample_count: Target number of samples (for lttb and simple modes)

    Returns:
        Tuple of (possibly downsampled) (x, y) arrays
    """
    if mode.lower() == 'full':
        return x, y
    elif mode.lower() == 'lttb':
        return lttb_downsample(x, y, sample_count)
    elif mode.lower() == 'simple':
        return simple_downsample(x, y, sample_count)
    else:
        return x, y
