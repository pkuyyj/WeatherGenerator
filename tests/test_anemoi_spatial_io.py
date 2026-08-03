# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

pytest.importorskip("anemoi.datasets")

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import ReaderData
from weathergen.datasets.healpix_domain import mask_to_contiguous_slices


class _FakeAnemoiDataset:
    def __init__(self, data):
        self.data = data
        self.variables = [f"var_{idx}" for idx in range(data.shape[1])]
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(index)
        return self.data[index]


def test_rank_local_read_selects_points_before_materializing_data():
    stored = np.arange(4 * 3 * 1 * 9, dtype=np.float64).reshape(4, 3, 1, 9)
    dataset = _FakeAnemoiDataset(stored)
    mask = np.array([False, True, True, False, True, False, True, True, True])
    reader = DataReaderAnemoi.__new__(DataReaderAnemoi)
    reader.ds = dataset
    reader.source_spatial_mask = mask
    reader.source_spatial_slices = mask_to_contiguous_slices(mask)

    actual = reader._read_data(1, 3, mask)

    expected = stored[1:3, :, 0][:, :, mask].astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert len(dataset.calls) == 3
    assert all(isinstance(call, tuple) and isinstance(call[-1], slice) for call in dataset.calls)


def test_source_hook_passes_spatial_mask_but_target_path_does_not(monkeypatch):
    reader = DataReaderAnemoi.__new__(DataReaderAnemoi)
    reader.source_idx = [1]
    reader.target_idx = [2]
    reader.source_spatial_mask = np.array([True, False])
    result = ReaderData.empty(1, 0)
    calls = []

    def fake_get(idx, channels_idx, spatial_mask=None):
        calls.append((idx, channels_idx, spatial_mask))
        return result

    monkeypatch.setattr(reader, "_get", fake_get)

    assert reader.get_source(4) is result
    assert reader.get_target(5) is result
    assert calls[0][0:2] == (4, [1])
    assert calls[0][2] is reader.source_spatial_mask
    assert calls[1] == (5, [2], None)


def test_spatial_subset_marker_survives_reader_filtering():
    reader_data = ReaderData(
        coords=np.array([[0.0, 0.0], [np.nan, 1.0]]),
        geoinfos=np.empty((2, 0)),
        data=np.ones((2, 1)),
        datetimes=np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[D]"),
        is_spatial_subset=True,
    )

    filtered = reader_data.remove_nan_coords_and_geoinfos()

    assert filtered.len() == 1
    assert filtered.is_spatial_subset
