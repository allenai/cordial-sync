from typing import Dict, Sequence, Tuple

import numpy as np

import constants


def manhattan_dists_between_positions(
    positions: Sequence[Dict[str, float]], grid_size: float
):
    dists_in_steps = [[] for _ in range(len(positions))]
    for i in range(len(positions) - 1):
        p0 = positions[i]
        for j in range(i + 1, len(positions)):
            p1 = positions[j]
            dist = int(
                round((abs(p0["x"] - p1["x"]) + abs(p0["z"] - p1["z"])) / grid_size)
            )
            dists_in_steps[i].append(dist)
            dists_in_steps[j].append(dist)
    return dists_in_steps


def pad_matrix_to_size_center(
    matrix: np.ndarray,
    desired_output_shape: Tuple[int, int],
    point_to_element_map: Dict[Tuple[float, float], Tuple[int, int]],
):
    assert matrix.shape[0] <= desired_output_shape[0]
    assert matrix.shape[1] <= desired_output_shape[1]
    pad_row = desired_output_shape[0] - matrix.shape[0]
    pad_col = desired_output_shape[1] - matrix.shape[1]
    pad_top = pad_row // 2
    pad_bottom = pad_row - pad_top
    pad_left = pad_col // 2
    pad_right = pad_col - pad_left
    pad_matrix = np.full(desired_output_shape, fill_value=constants.NO_INFO_SYM)
    assert pad_top + pad_bottom + matrix.shape[0] == desired_output_shape[0]
    assert pad_left + pad_right + matrix.shape[1] == desired_output_shape[1]
    pad_matrix[
        pad_top : pad_top + matrix.shape[0], pad_left : pad_left + matrix.shape[1]
    ] = matrix
    # update point to element map as per the padding
    point_to_pad_element_map = dict()
    for key, value in point_to_element_map.items():
        point_to_pad_element_map[key] = (value[0] + pad_top, value[1] + pad_left)
    return pad_matrix, point_to_pad_element_map


def pad_matrix(
    matrix: np.ndarray,
    pad: int,
    point_to_element_map: Dict[Tuple[float, float], Tuple[int, int]],
):
    pad_matrix = np.full(
        [matrix.shape[0] + 2 * pad, matrix.shape[1] + 2 * pad],
        fill_value=constants.NO_INFO_SYM,
    )
    pad_matrix[pad : pad + matrix.shape[0], pad : pad + matrix.shape[1]] = matrix
    # update point to element map as per the padding
    point_to_pad_element_map = dict()
    for key, value in point_to_element_map.items():
        point_to_pad_element_map[key] = (value[0] + pad, value[1] + pad)
    return pad_matrix, point_to_pad_element_map
