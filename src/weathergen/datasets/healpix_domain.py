# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import numpy as np
from astropy_healpix.healpy import ang2pix


@dataclass(frozen=True)
class HealpixDomain:
    """A consecutive range of nested HEALPix cells owned by one rank."""

    level: int
    cell_start: int
    cell_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.level, int | np.integer) or self.level < 0:
            raise ValueError(f"HEALPix level must be a non-negative integer, got {self.level!r}")
        num_cells = 12 * 4**self.level
        if not 0 <= self.cell_start < self.cell_end <= num_cells:
            raise ValueError(
                f"invalid HEALPix cell range [{self.cell_start}, {self.cell_end}) "
                f"for {num_cells} cells"
            )

    @property
    def num_cells(self) -> int:
        """Return the total number of cells at this HEALPix level."""
        return 12 * 4**self.level


def build_healpix_domain_mask(
    coords: np.typing.NDArray[np.floating],
    domain: HealpixDomain,
) -> np.typing.NDArray[np.bool_]:
    """Return a point mask for a rank-local HEALPix domain.

    ``coords`` use the reader convention ``(latitude, longitude)`` in degrees.
    The conversion intentionally matches the tokenizer's nested HEALPix mapping.
    """

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (num_points, 2), got {coords.shape}")

    valid = np.isfinite(coords).all(axis=1)
    mask = np.zeros(coords.shape[0], dtype=bool)
    if not valid.any():
        return mask

    thetas = np.deg2rad(90.0 - coords[valid, 0])
    phis = np.deg2rad(coords[valid, 1] + 180.0)
    cell_ids = ang2pix(2**domain.level, thetas, phis, nest=True)
    mask[valid] = (cell_ids >= domain.cell_start) & (cell_ids < domain.cell_end)
    return mask


def mask_to_contiguous_slices(mask: np.typing.NDArray[np.bool_]) -> list[slice]:
    """Convert a one-dimensional point mask to ordered contiguous read slices."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {mask.shape}")

    point_idxs = np.flatnonzero(mask)
    if point_idxs.size == 0:
        return []

    split_offsets = np.flatnonzero(np.diff(point_idxs) != 1) + 1
    runs = np.split(point_idxs, split_offsets)
    return [slice(int(run[0]), int(run[-1]) + 1) for run in runs]


def build_local_healpix_cell_splits(
    cell_ids: np.typing.NDArray[np.integer],
    num_cells: int,
    cell_start: int,
    cell_end: int,
) -> list[np.typing.NDArray[np.int64]]:
    """Group original point indices for one consecutive HEALPix-cell domain."""

    if not 0 <= cell_start < cell_end <= num_cells:
        raise ValueError(
            f"invalid HEALPix cell range [{cell_start}, {cell_end}) for {num_cells} cells"
        )

    # Domain-parallel filtering mask: this is applied independently to every
    # stream immediately after its coordinates have been mapped to nested
    # HEALPix cell IDs.
    local_domain_mask = (cell_ids >= cell_start) & (cell_ids < cell_end)
    local_point_idxs = np.flatnonzero(local_domain_mask)
    local_cell_ids = cell_ids[local_point_idxs]
    cell_splits = [np.array([], dtype=np.int64) for _ in range(cell_end - cell_start)]
    if local_point_idxs.size == 0:
        return cell_splits

    stable_args = {"stable": True} if int(np.__version__.split(".")[0]) >= 2 else {}
    local_order = np.argsort(local_cell_ids, **stable_args)
    sorted_point_idxs = local_point_idxs[local_order]
    sorted_cell_ids = local_cell_ids[local_order]
    split_offsets = np.flatnonzero(np.diff(sorted_cell_ids))
    point_idxs_by_occupied_cell = np.split(sorted_point_idxs, split_offsets + 1)

    for cell_id, point_idxs in zip(
        np.unique(sorted_cell_ids),
        point_idxs_by_occupied_cell,
        strict=True,
    ):
        cell_splits[cell_id - cell_start] = point_idxs

    return cell_splits
