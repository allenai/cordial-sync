from __future__ import division

import glob
import itertools
import json
import logging
import math
import os
import re
import shutil
import subprocess
from typing import Optional, Tuple, Sequence, Dict, Union

import numpy as np
import torch

import constants

try:
    from reprlib import repr
except ImportError:
    pass
from threading import Thread
from queue import Queue, Empty


def pad_matrix_to_size_topleft(
    matrix: np.ndarray,
    desired_output_shape: Tuple[int, int],
    point_to_element_map: Dict[Tuple[float, float], Tuple[int, int]],
    fill_value: Union[float, int] = constants.NO_INFO_SYM,
):
    assert matrix.shape[0] <= desired_output_shape[0]
    assert matrix.shape[1] <= desired_output_shape[1]
    pad_matrix = np.full(desired_output_shape, fill_value=fill_value)
    pad_matrix[0 : matrix.shape[0], 0 : matrix.shape[1]] = matrix
    return pad_matrix, point_to_element_map


def _joint_probability_tensor_from_mixture_slow(
    mixture_weights: torch.FloatTensor, marginal_prob_matrices: torch.FloatTensor
):
    """Used to error check joint_probability_tensor_from_mixture."""
    return sum(
        [
            mixture_weights[j]
            * outer_product(
                [
                    marginal_prob_matrices[i][j]
                    for i in range(marginal_prob_matrices.shape[0])
                ]
            )
            for j in range(marginal_prob_matrices.shape[1])
        ]
    )


def joint_probability_tensor_from_mixture(
    mixture_weights: torch.FloatTensor, marginal_prob_matrices: torch.FloatTensor
):
    assert len(mixture_weights.shape) == 1

    if mixture_weights.shape[0] == 2:
        v0 = marginal_prob_matrices[0].permute(1, 0)
        u0 = marginal_prob_matrices[1] * mixture_weights.view(-1, 1)
        return torch.matmul(v0, u0)

    product: Optional[torch.Tensor] = None
    new_shape = [mixture_weights.shape[0]] + [1] * marginal_prob_matrices.shape[0]
    for i, matrix in enumerate(marginal_prob_matrices):
        assert len(matrix.shape) == 2

        if i == 0:
            product = mixture_weights.reshape(*new_shape)
        else:
            new_shape[i] = 1
        new_shape[i + 1] = -1
        product = product * matrix.view(*new_shape)

    return product.sum(0)  # type: ignore


def joint_log_probability_tensor_from_mixture(
    log_mixture_weights: torch.FloatTensor,
    marginal_log_prob_matrices: torch.FloatTensor,
):
    assert len(log_mixture_weights.shape) == 1

    log_sum_tensor: Optional[torch.Tensor] = None
    new_shape = [log_mixture_weights.shape[0]] + [1] * marginal_log_prob_matrices.shape[
        0
    ]
    for i, matrix in enumerate(marginal_log_prob_matrices):
        assert len(matrix.shape) == 2

        if i == 0:
            log_sum_tensor = log_mixture_weights.reshape(*new_shape)
        else:
            new_shape[i] = 1
        new_shape[i + 1] = -1
        log_sum_tensor = log_sum_tensor + matrix.view(*new_shape)

    return log_sum_exp(log_sum_tensor, dim=0)  # type: ignore


def outer_product(vectors: Sequence[torch.FloatTensor]) -> torch.FloatTensor:
    assert len(vectors) > 1

    product: Optional[torch.Tensor] = None
    new_shape = [1] * len(vectors)
    for i, vector in enumerate(vectors):
        new_shape[i] = -1
        if i > 0:
            new_shape[i - 1] = 1
            product = product * vector.view(*new_shape)
        else:
            product = vector.view(*new_shape)

    return product  # type: ignore


def outer_sum(vectors: Sequence[torch.FloatTensor]) -> torch.FloatTensor:
    assert len(vectors) > 1

    sum_tensor: Optional[torch.Tensor] = None
    new_shape = [1] * len(vectors)
    for i, vector in enumerate(vectors):
        new_shape[i] = -1
        if i > 0:
            new_shape[i - 1] = 1
            sum_tensor = sum_tensor + vector.view(*new_shape)
        else:
            sum_tensor = vector.view(*new_shape)

    return sum_tensor  # type: ignore


def huber_loss(diff, delta):
    sq = diff.pow(2)
    abs = diff.abs()
    where_abs = (abs - delta >= 0).float()

    return (sq * (1.0 - where_abs) + (2 * delta * abs - delta ** 2) * where_abs).sum()


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()

    Taken from https://github.com/pytorch/pytorch/issues/2591
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        from numbers import Number

        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def fibonacci_sphere(samples=1):
    rnd = 1.0

    offset = 2.0 / samples
    increment = math.pi * (3.0 - math.sqrt(5.0))

    xs = []
    ys = []
    zs = []
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return {"xs": xs, "ys": ys, "zs": zs}


def save_project_state_in_log(
    call,
    task,
    local_start_time_str,
    dependent_data_paths: Optional[Tuple[str, ...]] = None,
    log_dir: str = "./logs/",
):
    short_sha = (
        subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    )
    log_file_path = os.path.join(log_dir, task, local_start_time_str)
    diff_path = os.path.join(log_file_path, "git-diff.txt")
    sha_path = os.path.join(log_file_path, "sha.txt")
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    if os.path.exists(diff_path):
        raise Exception("Diff should not already exist.")
    with open(diff_path, "w") as f:
        f.write(subprocess.check_output(["git", "diff"]).strip().decode("utf-8"))
    with open(sha_path, "w") as f:
        f.write(short_sha)

    # Save data that we are dependent on (e.g. previously trained models)
    if dependent_data_paths is not None:
        for path in dependent_data_paths:
            if path is not None:
                hash = get_hash_of_file(path)
                new_path = os.path.join(log_dir, "saved_data", hash + ".dat")
                if not os.path.exists(new_path):
                    shutil.copyfile(path, new_path)
                with open(
                    os.path.join(log_file_path, "saved_files_to_hashes.txt"), "a"
                ) as f:
                    f.write("{}\t{}\n".format(path, hash))

    # Finally save the call made to main
    with open(os.path.join(log_file_path, "call.json"), "w") as f:
        json.dump(call, f)

    return log_file_path


def random_orthogonal_matrix(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
        mat = np.eye(dim)
        mat[n - 1 :, n - 1 :] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def partition(n, num_parts):
    m = n // num_parts
    parts = [m] * num_parts
    num_extra = n % num_parts
    for i in range(num_extra):
        parts[i] += 1
    return parts


def expand_to_shape(shape, grid):
    new = np.zeros(shape=(shape[0], shape[1]))
    row_parts = np.cumsum(partition(shape[0], grid.shape[0]))
    col_parts = np.cumsum(partition(shape[1], grid.shape[1]))
    for i in range(grid.shape[0]):
        if i == 0:
            r0, r1 = (0, row_parts[i])
        else:
            r0, r1 = (row_parts[i - 1], row_parts[i])

        for j in range(grid.shape[1]):
            if j == 0:
                c0, c1 = (0, col_parts[j])
            else:
                c0, c1 = (col_parts[j - 1], col_parts[j])

            new[r0:r1, c0:c1] = grid[i, j]
    return new


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    dir = "/".join(log_file.split("/")[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    fileHandler = logging.FileHandler(log_file, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, "r"))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    sentinal = object()
    for param, shared_param in itertools.zip_longest(
        model.parameters(), shared_model.parameters(), fillvalue=sentinal
    ):
        assert sentinal is not param and sentinal is not shared_param
        if shared_param.requires_grad:
            assert param.requires_grad
            if not gpu or param.grad is None:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def get_hash_of_file(file_path):
    import hashlib, os

    sha_hash = hashlib.sha1()
    if not os.path.exists(file_path):
        raise IOError("File " + file_path + " not found.")

    with open(file_path, "rb") as f:
        while True:
            # Read file in as little chunks
            buf = f.read(4096)
            if not buf:
                break
            sha_hash.update(hashlib.sha1(buf).hexdigest().encode("utf-8"))

    return sha_hash.hexdigest()


def get_hash_of_dirs(directory, verbose=0):
    # http://akiscode.com/articles/sha-1directoryhash.shtml
    # Copyright (c) 2009 Stephen Akiki
    # MIT License (Means you can do whatever you want with this)
    #  See http://www.opensource.org/licenses/mit-license.php
    # Error Codes:
    #   -1 -> Directory does not exist
    #   -2 -> General error (see stack traceback)
    import hashlib, os

    sha_hash = hashlib.sha1()
    if not os.path.exists(directory):
        return -1

    try:
        for root, dirs, files in os.walk(directory):
            for names in files:
                if verbose == 1:
                    print("Hashing", names)
                filepath = os.path.join(root, names)
                try:
                    f1 = open(filepath, "rb")
                except Exception as e:
                    # You can't open the file for some reason
                    f1.close()
                    continue

                while True:
                    # Read file in as little chunks
                    buf = f1.read(4096)
                    if not buf:
                        break
                    sha_hash.update(hashlib.sha1(buf).hexdigest().encode("utf-8"))
                f1.close()
    except Exception as e:
        raise e

    return sha_hash.hexdigest()


def round_to_factor(num, base) -> int:
    return int((num / base)) * base


def key_for_point(x, z):
    return "%0.1f %0.1f" % (x, z)


def point_for_key(key):
    x, z = key.split("|")
    return dict(x=x, z=z)


def location_to_metadata(loc):
    assert "x" in loc.keys()
    assert "y" in loc.keys()
    assert "z" in loc.keys()
    assert "rotation" in loc.keys()
    assert "horizon" in loc.keys()
    meta = dict()
    meta["position"] = dict(x=loc["x"], y=loc["y"], z=loc["z"])
    meta["rotation"] = dict(y=round(loc["rotation"]))
    meta["cameraHorizon"] = round(loc["horizon"])
    return meta


def models_with_log_name(log_name, model_folder):
    exp_name, date_time = log_name.split("/")
    model_names_all = glob.glob(
        os.path.join(model_folder, exp_name + "_*_" + date_time + ".dat")
    )
    model_names = []
    model_iterations = []
    for name in model_names_all:
        search_string = exp_name + "_" + "(.*)" + "_" + date_time + ".dat"
        iter = int(re.search(search_string, name).group(1))
        if iter % 10000 == 0:
            model_names.append(name)
            model_iterations.append(iter)
    sorted_model_names = [x for _, x in sorted(zip(model_iterations, model_names))]
    sorted_model_iterations = [y for y, _ in sorted(zip(model_iterations, model_names))]
    return sorted_model_names, sorted_model_iterations


def last_model_with_log_name(log_name, model_folder):
    sorted_model_names, sorted_model_iterations = models_with_log_name(
        log_name, model_folder
    )
    assert len(sorted_model_names) >= 1
    return sorted_model_names[-1], sorted_model_iterations[-1]


def first_model_with_log_name(log_name, model_folder):
    sorted_model_names, sorted_model_iterations = models_with_log_name(
        log_name, model_folder
    )
    assert len(sorted_model_names) >= 1
    return sorted_model_names[0], sorted_model_iterations[0]


def manhattan_dist_between_two_positions(p0, p1):
    dist = int(round((abs(p0["x"] - p1["x"]) + abs(p0["z"] - p1["z"])) / 0.25))
    return dist


class NonBlockingStreamReader:
    # Taken from http://eyalarubas.com/python-subproc-nonblock.html
    def __init__(self, stream):
        """
        stream: the stream to read from.
                Usually a process' stdout or stderr.
        """

        self._s = stream
        self._q = Queue()

        def _populateQueue(stream, queue):
            """
            Collect lines from 'stream' and put them in 'quque'.
            """

            while True:
                line = stream.readline()
                if line:
                    queue.put(line)
                else:
                    break
                    # raise UnexpectedEndOfStream

        self._t = Thread(target=_populateQueue, args=(self._s, self._q))
        self._t.daemon = True
        self._t.start()  # start collecting lines from the stream

    def readline(self, timeout=None):
        try:
            return self._q.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None


def all_equal(seq: Sequence):
    if len(seq) <= 1:
        return True
    return all(seq[0] == s for s in seq[1:])


def unzip(xs):
    a = None
    n = None
    for x in xs:
        if n is None:
            n = len(x)
            a = [[] for _ in range(n)]
        for i, y in enumerate(x):
            a[i].append(y)
    return a
