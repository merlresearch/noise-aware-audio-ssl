# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List, Optional, Tuple

import numpy as np


def random_split_path(
    path_list: List[Any],
    train_valid_split_ratio: Optional[List[float]] = None,
    train_valid_test_split_count: Optional[List[int]] = None,
    seed: int = 0,
) -> Tuple[List[Any], List[Any], List[Any]]:
    # Use numpy to avoid in-place shuffling
    rng = np.random.default_rng(seed)
    path_list = rng.permutation(path_list).tolist()
    total = len(path_list)
    if train_valid_split_ratio is not None:
        assert train_valid_test_split_count is None
        assert sum(train_valid_split_ratio) <= 1.0
        train_count = int(total * train_valid_split_ratio[0])
        valid_count = int(total * train_valid_split_ratio[1])
    elif train_valid_test_split_count is not None:
        assert train_valid_split_ratio is None
        train_count, valid_count, test_count = train_valid_test_split_count
        assert train_count + valid_count + test_count == total
    else:
        raise ValueError("Either train_valid_split_ratio or train_valid_test_split_count must be provided.")

    train_paths = path_list[:train_count]
    valid_paths = path_list[train_count : train_count + valid_count]
    test_paths = path_list[train_count + valid_count :]
    return train_paths, valid_paths, test_paths
