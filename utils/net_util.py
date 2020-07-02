"""Contains a bunch of utilities useful during network training in PyTorch."""
import math
from collections import deque
from typing import Dict, Union, List, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def init_orthogonal(tensor, gain=1):
    r"""
    Taken from a future torch version
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


def get_gpu_memory_map():
    # From https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    import subprocess

    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def load_model_from_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    dict_keys = set(state_dict.keys())

    in_model_not_in_dict = []
    in_dict_not_in_model = []
    wrong_parameter_sizes = []
    for key in model_keys | dict_keys:
        if key in model_keys and key not in dict_keys:
            in_model_not_in_dict.append(key)
        elif key not in model_keys and key in dict_keys:
            in_dict_not_in_model.append(key)
        elif model_state_dict[key].shape != state_dict[key].shape:
            wrong_parameter_sizes.append(key)

    if (
        len(in_model_not_in_dict) == 0
        and len(in_dict_not_in_model) == 0
        and len(wrong_parameter_sizes) == 0
    ):
        return model.load_state_dict(state_dict)
    else:
        print(
            (
                "WARNING: Loading model from state dictionary but:\n"
                "* The following parameters are present in the state"
                " dictionary and not in the model and will be ignored: {}\n"
                "* The following parameters are present in the model but "
                "not in the state and will remain in their initial state: {}\n"
                "* The following parameters are present in both the model and "
                "saved state but are of incompatible sizes, they will remain as in the model: {}\n"
            ).format(
                "\n\t- None"
                if len(in_dict_not_in_model) == 0
                else "\n\t- " + "\n\t- ".join(in_dict_not_in_model),
                "\n\t- None"
                if len(in_model_not_in_dict) == 0
                else "\n\t- " + "\n\t- ".join(in_model_not_in_dict),
                "\n\t- None"
                if len(wrong_parameter_sizes) == 0
                else "\n\t- " + "\n\t- ".join(wrong_parameter_sizes),
            )
        )
        yn = input("Continue? (y/n)").lower().strip()
        if yn not in ["y", "yes"]:
            print("Aborting...")
            quit()

        return model.load_state_dict(
            {
                **model.state_dict(),
                **{
                    k: state_dict[k]
                    for k in ((dict_keys - set(wrong_parameter_sizes)) & model_keys)
                },
            }
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_search_for_model_with_least_upper_bound_parameters(
    target_parameter_count,
    create_model_func: Callable[[int], nn.Module],
    lower: int,
    upper: int,
):
    assert lower <= upper

    lower_count = count_parameters(create_model_func(lower))
    upper_count = count_parameters(create_model_func(upper))

    assert lower_count <= target_parameter_count <= upper_count, "Invalid range"

    def run_search(
        target_parameter_count,
        create_model_func: Callable[[int], nn.Module],
        lower: int,
        upper: int,
    ):
        if lower == upper:
            return lower

        mid = int(math.floor((lower + upper) / 2))
        mid_count = count_parameters(create_model_func(mid))

        if mid_count == target_parameter_count:
            return mid
        elif mid_count > target_parameter_count:
            return run_search(
                target_parameter_count=target_parameter_count,
                create_model_func=create_model_func,
                lower=lower,
                upper=mid,
            )
        else:
            return run_search(
                target_parameter_count=target_parameter_count,
                create_model_func=create_model_func,
                lower=mid + 1,
                upper=upper,
            )

    return run_search(
        target_parameter_count=target_parameter_count,
        create_model_func=create_model_func,
        lower=lower,
        upper=upper,
    )


def logit_offsets_for_conditional_probabilities(
    action_group_dims: Tuple[int, ...]
) -> List[float]:
    consts = [0.0]
    for i in range(1, len(action_group_dims)):
        consts.append(math.log(action_group_dims[0] / action_group_dims[i]))
    return consts


class ScalarMeanTracker(object):
    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add_scalars(self, scalars: Dict[str, Union[float, int]]) -> None:
        for k in scalars:
            if np.isscalar(scalars[k]):
                if k not in self._sums:
                    self._sums[k] = float(scalars[k])
                    self._counts[k] = 1
                else:
                    self._sums[k] += float(scalars[k])
                    self._counts[k] += 1

    def means(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        return means

    def counts(self):
        return {**self._counts}

    def pop_and_reset_for_key(self, k):
        s = self._sums[k]
        c = self._counts[k]
        del self._sums[k]
        del self._counts[k]
        return s / c

    def pop_and_reset(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means


class TensorConcatTracker(object):
    def __init__(self) -> None:
        self._tensors: Dict[str, torch.FloatTensor] = {}

    def add_tensors(self, tensors: Dict[str, Union[torch.FloatTensor, Any]]) -> None:
        for k in tensors:
            if type(tensors[k]) == torch.FloatTensor:
                if k not in self._tensors:
                    self._tensors[k] = tensors[k]
                else:
                    self._tensors[k] = torch.cat((self._tensors[k], tensors[k]), dim=0)

    def pop_and_reset(self):
        t = self._tensors
        self._tensors = {}
        return t


class RollingAverage(object):
    """Computes and stores the running average as well
    as the average within a recent window"""

    def __init__(self, window_size):
        assert window_size > 0
        self.window_size = window_size
        self.rolling_sum = 0
        self.sum = 0
        self.count = 0
        self.rolling_deque = deque()

    def add(self, val):
        """Add one value."""
        self.sum += val
        self.rolling_sum += val
        self.count += 1
        self.rolling_deque.append(val)
        if len(self.rolling_deque) > self.window_size:
            self.rolling_sum -= self.rolling_deque.popleft()

    def rolling_average(self):
        assert self.count > 0
        return self.rolling_sum / (1.0 * len(self.rolling_deque))

    def full_average(self):
        assert self.count > 0
        return self.sum / self.count


class TrainTestInfoStore(object):
    def __init__(self, train_window_size, test_window_size):
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size

        self.train_recent_save = []
        self.train_averages = []
        self.train_num_frames = []
        self.test_recent_save = []
        self.test_averages = []
        self.test_num_frames = []

    def add_train_result(self, episode_reward, num_frames):
        self.train_recent_save.append(episode_reward)
        if len(self.train_recent_save) == self.train_window_size:
            self.train_averages.append(np.mean(self.train_recent_save))
            self.train_num_frames.append(num_frames)
            self.train_recent_save = []

    def add_test_result(self, episode_reward, num_frames):
        self.test_recent_save.append(episode_reward)
        if len(self.test_recent_save) == self.test_window_size:
            self.test_averages.append(np.mean(self.test_recent_save))
            self.test_num_frames.append(num_frames)
            self.test_recent_save = []

    def train_results(self):
        return self.train_averages, self.train_num_frames

    def test_results(self):
        return self.test_averages, self.test_num_frames

    def train_full_average(self):
        sum = (
            np.sum(self.train_recent_save)
            + np.sum(self.train_averages) * self.train_window_size
        )
        return sum / (
            len(self.train_averages) * self.train_window_size
            + len(self.train_recent_save)
        )

    def test_full_average(self):
        sum = (
            np.sum(self.test_recent_save)
            + np.sum(self.test_averages) * self.test_window_size
        )
        return sum / (
            len(self.test_averages) * self.test_window_size + len(self.test_recent_save)
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Resets counters."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates counters."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_gpu_model(model, optim, epoch, ckpt_fname):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    optimizer = optim.state_dict()
    for key in optimizer.keys():
        optimizer[key] = optimizer[key].cpu()

    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "optimizer": optimizer}, ckpt_fname
    )


def save_model(model, optim, epoch, ckpt_fname):
    state_dict = model.state_dict()
    optimizer = optim.state_dict()

    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "optimizer": optimizer}, ckpt_fname
    )


def show_image_stack(image_stack):
    """Displays the stack of images

    If the image_stack is of type torch.Tensor, then expected size is (1, N, H, W)
    If the image_stack is of type np.ndarray, then expected size is (H, W, N)
    """
    import matplotlib

    matplotlib.use("TkAgg", force=False)
    import matplotlib.pyplot as plt  # Keeping this outside causes issues in multiprocessing.

    if isinstance(image_stack, torch.Tensor):
        image_stack = image_stack.squeeze().cpu().numpy()
        image_stack = np.transpose(image_stack, (1, 2, 0))
    num_images = image_stack.shape[2]
    length = np.ceil(np.sqrt(num_images)).item()

    plt.figure()
    for idx in range(num_images):
        plt.subplot(length, length, idx + 1)
        img = image_stack[:, :, idx]
        plt.imshow(img, cmap="gray")
    plt.show()


def recursively_detach(to_detach: Any):
    """Recursively detach tensors in nested structure."""
    if to_detach is None:
        return to_detach
    elif isinstance(to_detach, tuple):
        return tuple(recursively_detach(x) for x in to_detach)
    elif isinstance(to_detach, list):
        return [recursively_detach(x) for x in to_detach]
    elif isinstance(to_detach, dict):
        return {k: recursively_detach(to_detach[k]) for k in to_detach}
    elif isinstance(to_detach, set):
        return set(recursively_detach(x) for x in to_detach)
    elif (
        isinstance(to_detach, np.ndarray)
        or np.isscalar(to_detach)
        or isinstance(to_detach, str)
    ):
        return to_detach
    elif isinstance(to_detach, torch.Tensor):
        return to_detach.detach()
    elif hasattr(to_detach, "repackage_hidden"):
        return to_detach.repackage_hidden()
    elif hasattr(to_detach, "detach"):
        return to_detach.detach()
    else:
        raise NotImplementedError(
            "Sorry, hidden state of type {} is not supported.".format(type(to_detach))
        )


def put_tensor_onto_gpu_of_template(tensor, template):
    if template.is_cuda:
        with torch.cuda.device(template.get_device()):
            tensor = tensor.cuda()
        return tensor
    else:
        return tensor.cpu()


def resnet_input_transform(input_image, im_size):
    """Takes in numpy ndarray of size (H, W, 3) and transforms into tensor for
       resnet input.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    all_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            ScaleBothSides(im_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transformed_image = all_transforms(input_image)
    return transformed_image


def resize_image(input_image, im_size):
    all_transforms = transforms.Compose(
        [transforms.ToPILImage(), ScaleBothSides(im_size), transforms.ToTensor()]
    )
    return all_transforms(input_image)


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


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


class ScaleBothSides(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of both edges, and this can change aspect ratio.
    size: output size of both edges
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)
