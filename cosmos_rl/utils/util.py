# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import ast
import multiprocessing
import json
import datasets
import socket
import queue
import dataclasses
import base64
import struct
import tarfile
import ctypes
import asyncio
import importlib
import sys
import glob
import io
from PIL import Image
from filelock import FileLock, Timeout
from collections import OrderedDict
from functools import wraps
from msgpack import ExtType
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
import torch
import pynvml
from contextlib import contextmanager
from torch.functional import F
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    snapshot_download,
    HfFileSystem,
)
import time
import functools
from cosmos_rl.utils.logging import logger
from safetensors import safe_open
from cosmos_rl.utils.constant import CACHE_DIR
import math


def create_cached_dir_if_needed():
    """
    Creates the local cached dir if it doesn't exist.
    """
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_data_from_disk_or_hf(data_name, data_subset=None, revision=None):
    if data_subset is not None and len(data_subset) == 0:
        data_subset = None

    if os.path.exists(data_name):
        try:
            return datasets.load_from_disk(data_name)
        except Exception as e:
            logger.warning(
                f"Failed to load dataset from disk: {e}. Trying to load from HuggingFace Hub..."
            )
            return datasets.load_dataset(data_name, data_subset, revision=revision)
    return datasets.load_dataset(data_name, data_subset, revision=revision)


def read_json_file(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def write_json_file(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def resolve_model_path(model_path: str, revision: Optional[str] = None) -> str:
    if not os.path.exists(model_path.replace(":", "/")):
        if ":" in model_path:
            if model_path.count(":") > 1:
                raise ValueError(
                    f"Invalid model path {model_path}, should be in the format of 'owner/repo[:path]'"
                )
            model_path, file_path = model_path.split(":")
        else:
            model_path = model_path
            file_path = None

        # Check whether `model_path` is a HuggingFace repo id with repo name
        if len(model_path.split("/")) == 2:
            logger.info(
                f"model path {model_path} is not a directory. Trying to load from HuggingFace Hub..."
            )

            hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN", None))
            files = hf_fs.ls(model_path, detail=False)
            if (
                os.path.join(model_path, "model.safetensors.index.json") in files
                or os.path.join(model_path, "model.safetensors") in files
            ):
                logger.info(
                    f"Found safetensors in {model_path}. Ignoring *pytorch_model* and *consolidated* files."
                )
                ignore_patterns = ["*pytorch_model*", "*consolidated*"]
            else:
                ignore_patterns = None

            try:
                cache_dir = os.environ.get("HF_HOME", "")
                if not cache_dir:
                    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers/")
                else:
                    cache_dir = os.path.join(cache_dir, "hub")
                model_path = retry(snapshot_download)(
                    model_path,
                    revision=revision,
                    token=os.environ.get("HF_TOKEN"),
                    cache_dir=cache_dir,
                    ignore_patterns=ignore_patterns,
                    allow_patterns=file_path,
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                raise

            if file_path is not None:
                model_path = os.path.join(model_path, file_path)
            logger.info(f"Downloaded model from HuggingFace to {model_path}")
        elif os.path.isdir(model_path):
            # In case model_path is a local directory, we need to check if the file_path exists
            if file_path is not None:
                model_path = os.path.join(model_path, file_path)
        else:
            raise ValueError(
                f"Model path {model_path} is not a directory and not a valid HuggingFace repo id with repo name."
            )
    else:
        model_path = model_path.replace(":", "/")
    return model_path


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def find_available_port(start_port):
    max_port = 65535  # Maximum port number
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError("No available ports found in the specified range.")


def put_with_overwrite(q: queue.Queue, item):
    assert isinstance(q, queue.Queue), "q must be a queue.Queue, but not asyncio.Queue"
    if q.full():
        try:
            q.get_nowait()  # Remove the oldest item
        except queue.Empty:
            pass  # In case the queue becomes empty in a race condition
    q.put(item)


def extract_fields(dc_instance):
    """
    Recursively extract dataclass fields into a nested dictionary.
    Leaf nodes contain a dict with 'value', 'metadata', 'type', and 'input_type'.
    """

    def recursive_extract(instance):
        fields = {}
        for f in dataclasses.fields(instance):
            if f.metadata.get("skip_ui", False):
                continue
            value = getattr(instance, f.name)

            # Special handling for train_policy field
            if f.name == "train_policy" and hasattr(instance, "train_policy"):
                fields[f.name] = recursive_extract(
                    value
                )  # Just extract current policy's fields
                continue

            if dataclasses.is_dataclass(value):
                fields[f.name] = recursive_extract(value)
            else:
                field_data = {
                    "value": value,
                    "metadata": f.metadata,
                    "type": f.type,
                    "input_type": "checkbox" if f.type == bool else "text",
                }
                fields[f.name] = field_data
        return fields

    return recursive_extract(dc_instance)


def parse_collection(s):
    """
    Attempts to parse a string into a Python literal.

    If the string represents a list or tuple, it returns the corresponding
    Python object. Otherwise, it returns None.
    """
    try:
        result = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # The string is not a valid Python literal.
        return None

    # Check if the result is a list or tuple.
    if isinstance(result, (list, tuple)):
        return result
    else:
        return None


def list_to_b64(lst) -> str:
    # for int64_t listy to base64
    byte_data = struct.pack(f"{len(lst)}q", *lst)
    return base64.b64encode(byte_data).decode("utf-8")


def b64_to_list(b64_str) -> List[int]:
    # for base64 to int64_t list
    byte_data = base64.b64decode(b64_str)
    n = len(byte_data) // 8
    return list(struct.unpack(f"{n}q", byte_data))


def str2torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string representation of a dtype to a torch.dtype.
    """
    dtype_str = dtype_str.lower()
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


@contextmanager
def cosmos_default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


class IdentityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@torch.compile
def _logsumexp_fp32_backward(grad_output, input, logsumexp_values):
    # grad_output shape: (...), need to unsqueeze to match input shape for broadcasting
    grad_output_expanded = grad_output.unsqueeze(-1)
    # Compute softmax: exp(input - logsumexp)
    softmax = (input - logsumexp_values.unsqueeze(-1)).exp()
    return (grad_output_expanded * softmax).to(input.dtype)


class LogSumExpFp32Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        logsumexp_values = torch.zeros(
            input.shape[:-1], dtype=torch.float32, device=input.device
        )
        torch.logsumexp(input, dim=-1, out=logsumexp_values)
        ctx.save_for_backward(input, logsumexp_values)
        ctx.input_dtype = input.dtype
        return (
            logsumexp_values.to(torch.bfloat16)
            if input.dtype == torch.bfloat16
            else logsumexp_values
        )

    @staticmethod
    def backward(ctx, grad_output):
        input, logsumexp_values = ctx.saved_tensors
        return _logsumexp_fp32_backward(grad_output, input, logsumexp_values)


def _logsumexp_fp32(input):
    """
    Computes the logsumexp of the input tensor in float32 precision (even if input is Bf16).
    """
    return LogSumExpFp32Backward.apply(input)


_SELECTIVE_LOG_SOFTMAX_OPTIM = False


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if _SELECTIVE_LOG_SOFTMAX_OPTIM:
        # NOTE: aazzolini optimized.
        selected_logits = (
            torch.gather(logits, dim=-1, index=index.unsqueeze(-1))
            .squeeze(-1)
            .to(torch.float32)
        )
        logsumexp_values = _logsumexp_fp32(logits)
        return (selected_logits - logsumexp_values).to(
            logits.dtype
        )  # log_softmax(x_i) = x_i - logsumexp(x)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def clear_weight_name(name: str) -> str:
    return name.replace("._orig_mod", "").replace("._checkpoint_wrapped_module", "")


def _extract_tgz_file(tgz_file, output_dir):
    with tarfile.open(tgz_file, "r:gz") as tar:
        tar.extractall(path=output_dir)


def basename_from_modelpath(path: str) -> str:
    path = path.strip().strip("/")
    if len(path.split("/")) > 2:
        path = "/".join(path.split("/")[-2:])
    return path


def if_use_modelscope(path: str) -> bool:
    modelscope_cache_dir = os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache/modelscope_cache/dataset"),
    )
    return path.startswith(modelscope_cache_dir)


def prepare_cosmos_data(dataset, *args, **kwargs):
    cache_dir = os.environ.get(
        "COSMOS_CACHE", os.path.join(os.path.expanduser("~"), ".cache/cosmos/")
    )
    dataset_name = basename_from_modelpath(dataset.name)
    use_modelscope = if_use_modelscope(dataset.name)
    dataset_dir = os.path.join(
        cache_dir,
        "datasets",
        dataset_name,
        dataset.subset,
    )
    video_clips_dir = os.path.join(dataset_dir, "video_clips")

    prepared_flag = os.path.join(video_clips_dir, ".prepared")
    if os.path.exists(prepared_flag):
        logger.info(f"Dataset {dataset.name} already prepared.")
        return

    # ensure base dirs exist
    os.makedirs(video_clips_dir, exist_ok=True)

    # file-lock to prevent races
    lock_path = os.path.join(dataset_dir, "prepare.lock")
    lock = FileLock(lock_path, timeout=1800)  # wait up to 30 minutes

    try:
        with lock:
            # re-check sentinel inside the lock
            if os.path.exists(prepared_flag):
                logger.info(
                    f"Dataset {dataset.name} already prepared by another worker."
                )
                return

            # --- find .tar.gz clips ---
            re_pattern = re.compile(rf"^{re.escape(dataset.subset)}/clips/.*\.tar\.gz$")
            file_pattern = f"{dataset.subset}/clips/*.tar.gz"
            if use_modelscope:
                assert os.path.exists(dataset.name), f"{dataset.name} not found locally"
                remote_files = [
                    f.replace(dataset.name + "/", "")
                    for f in glob.glob(f"{dataset.name}/**/*.tar.gz", recursive=True)
                ]
            else:
                remote_files = list_repo_files(
                    repo_id=dataset_name,
                    repo_type="dataset",
                    revision=dataset.revision or None,
                )

            tgz_files = [f for f in remote_files if re_pattern.match(f)]

            # --- extract clips ---
            if tgz_files:
                if use_modelscope:
                    downloaded_clips_dir = os.path.join(
                        dataset.name, dataset.subset, "clips"
                    )
                else:
                    snapshot_dir = retry(snapshot_download)(
                        dataset_name,
                        allow_patterns=[file_pattern],
                        repo_type="dataset",
                        revision=dataset.revision or None,
                    )
                    downloaded_clips_dir = os.path.join(
                        snapshot_dir, dataset.subset, "clips"
                    )

                assert os.path.exists(
                    downloaded_clips_dir
                ), f"Cannot find clips at {downloaded_clips_dir}"

                # parallel extract
                results = {}
                with tqdm(total=len(tgz_files), desc="Extracting clips") as pbar:
                    with multiprocessing.Pool(
                        processes=min(multiprocessing.cpu_count(), 8)
                    ) as pool:
                        for tgz in tgz_files:
                            full_path = os.path.join(
                                downloaded_clips_dir, os.path.basename(tgz)
                            )
                            results[tgz] = pool.apply_async(
                                _extract_tgz_file,
                                (full_path, video_clips_dir),
                                callback=lambda _: pbar.update(1),
                            )
                        pool.close()
                        pool.join()

                # check for extract errors
                for tgz, res in results.items():
                    if not res.successful():
                        raise RuntimeError(f"Failed to extract {tgz}: {res.get()}")

            else:
                # legacy single-archive format
                clip_tgz = os.path.join(dataset.subset, "clips.tar.gz")
                if use_modelscope:
                    clip_tgz = os.path.join(dataset.name, clip_tgz)
                else:
                    clip_tgz = hf_hub_download(
                        repo_id=dataset_name,
                        revision=dataset.revision or None,
                        repo_type="dataset",
                        filename=clip_tgz,
                    )
                _extract_tgz_file(clip_tgz, video_clips_dir)

            # --- special renaming for 'av' subset ---
            if dataset.subset == "av":
                for root, _, files in os.walk(video_clips_dir):
                    for file in files:
                        if file.endswith(".mp4"):
                            src = os.path.join(root, file)
                            dst = os.path.join(root, file.split(".")[0] + ".mp4")
                            os.rename(src, dst)
                            logger.info(f"Renamed {file} → {os.path.basename(dst)}")

            # finally, create our sentinel
            open(prepared_flag, "w").close()
            logger.info(f"Finished preparing {dataset.name}.")

    except Timeout:
        raise RuntimeError(
            f"Timeout acquiring lock for {dataset.name}; another process may be stuck."
        )


GPU_FLOPS_MAPPING = {
    "NVIDIA H100 80GB HBM3": {
        "FP32": 989 * (1e12),
        "FP16": 1979 * (1e12),
    },
    "NVIDIA A100 80GB": {
        "FP32": 39 * (1e12),
        "FP16": 195 * (1e12),
    },
}


def get_device_flops(dtype: torch.dtype, num_gpus: int) -> int:
    """
    Get the GPU FLOPs for the current device.

    Args:
        dtype (torch.dtype): The data type of the model.
        num_gpus (int): The number of GPUs available.

    Returns:
        int: The FLOPs of the current device.
    """
    gpu_flops = 0
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_type = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        if dtype == "float32":
            gpu_flops = GPU_FLOPS_MAPPING[gpu_type]["FP32"]
        elif dtype == "float16" or dtype == "bfloat16":
            gpu_flops = GPU_FLOPS_MAPPING[gpu_type]["FP16"]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    else:
        logger.warning("CUDA is not available. Cannot get GPU FLOPs.")
    return gpu_flops * num_gpus


def compute_mfu(
    model: torch.nn.Module,
    n_tokens: int,
    iter_time: float,
    num_gpus: int,
    dtype: str,
) -> dict:
    """
    Compute the model FLOPs Utilization (MFU) for a given model and number of tokens.

    Args:
        model (torch.nn.Module): The model for which to compute the MFU.
        n_tokens (int): The number of tokens in the inputs.
        iter_time (float): The time taken for the forward pass in seconds.

    Returns:
        dict: A dictionary containing the MFU/FLOPs value.
    """
    result = {}
    # Get the model FLOPs for the iteration
    model_params, model_flops = model.get_nparams_and_flops(n_tokens)
    result["model_flops"] = model_flops

    # Get the GPU FLOPs
    try:
        gpu_flops = get_device_flops(dtype, num_gpus)
        # Calculate and return the MFU
        mfu = (model_flops / gpu_flops) / iter_time
        result["mfu"] = mfu
        logger.info(
            f"MFU: {mfu:.4f}, Model FLOPs: {model_flops / 1e12:.2f}T, GPU FLOPs: {gpu_flops / 1e12:.2f}T, Iter time: {iter_time:.2f}s"
        )
    except Exception as e:
        logger.error(
            f"Cannot compute MFU: {e}, only report model FLOPs, you can calculate MFU manually."
        )
        logger.info(
            f"Model FLOPs: {model_flops / 1e12:.2f}T, Iter time: {iter_time:.2f}s"
        )

    return result


def fix_data_type_size(obj):
    if isinstance(obj, tuple):
        return tuple([fix_data_type_size(x) for x in obj])
    elif isinstance(obj, list):
        return [fix_data_type_size(x) for x in obj]
    elif isinstance(obj, dict):
        return {fix_data_type_size(k): fix_data_type_size(v) for k, v in obj.items()}
    elif isinstance(obj, int):
        return ctypes.c_int64(obj)
    else:
        return obj


# Extension code
MSGPACK_C_LONG_EXT_TYPE = 0x10


# Hooks
def msgpack_c_long(obj):
    if isinstance(obj, ctypes._SimpleCData) and isinstance(obj.value, int):
        if isinstance(obj, ctypes.c_long):
            return ExtType(
                MSGPACK_C_LONG_EXT_TYPE, obj.value.to_bytes(8, "big", signed=True)
            )
    raise TypeError(f"Unsupported type: {type(obj)}")


def msgunpack_c_long(code, data):
    if code == MSGPACK_C_LONG_EXT_TYPE:
        val = int.from_bytes(data, "big", signed=True)
        return int(val)
    return ExtType(code, data)


def sync_model_vocab(
    model_name_or_path,
    lm_head_key="lm_head.weight",
    embed_tokens_key="model.embed_tokens.weight",
):
    self_rank = int(os.environ.get("RANK", 0))
    vocab_size = None
    if self_rank == 0:
        model_index_path = f"{model_name_or_path}:model.safetensors.index.json"
        weight_map_path = resolve_model_path(model_index_path)
        has_weight_map = os.path.exists(weight_map_path)
        # Check if the weight map file exists
        if has_weight_map:
            weight_map = read_json_file(weight_map_path)["weight_map"]
            possible_prefix = ["", "model.", "language_model."]
            possible_keys = [prefix + lm_head_key for prefix in possible_prefix] + [
                prefix + embed_tokens_key for prefix in possible_prefix
            ]
            for key in possible_keys:
                if key in weight_map:
                    head_or_embed_tokens = weight_map[key]
                    head_or_embed_tokens_path = resolve_model_path(
                        f"{model_name_or_path}:{head_or_embed_tokens}"
                    )
                    with safe_open(
                        head_or_embed_tokens_path, framework="pt", device="cpu"
                    ) as f:
                        tensor_slice = f.get_slice(key)
                    vocab_size, _ = tensor_slice.get_shape()
                    break
        else:
            # models like google/gemma-3-1b-pt does not have model.safetensors.index.json
            model_safetensors_path = resolve_model_path(
                f"{model_name_or_path}:model.safetensors"
            )
            with safe_open(model_safetensors_path, framework="pt", device="cpu") as f:
                tensor_names = f.keys()
                possible_prefix = ["", "model.", "language_model."]
                possible_keys = [prefix + lm_head_key for prefix in possible_prefix] + [
                    prefix + embed_tokens_key for prefix in possible_prefix
                ]
                for key in possible_keys:
                    if key in tensor_names:
                        tensor_slice = f.get_slice(key)
                        vocab_size, _ = tensor_slice.get_shape()
                        break
        if vocab_size is None:
            raise ValueError(
                "Could not find `lm_head` or `model.embed_tokens.weight` in the model."
            )

    from cosmos_rl.utils.distributed import broadcast_object_cpu

    vocab_size = broadcast_object_cpu(vocab_size, src=0, device=torch.device("cpu"))
    logger.info(f"Vocabulary size: {vocab_size}")

    return vocab_size


def retry(func=None, *, max_retry=10, max_delay=30.0):
    """
    Decorator (or wrapper) to retry a function up to max_retry times,
    backing off exponentially (1s, 2s, 4s, …) up to max_delay seconds.

    Usage:

      @retry(max_retry=5)                # uses default max_delay=30
      def foo(...): …

      @retry(max_retry=5, max_delay=60)  # override max_delay
      def bar(...): …

      wrapped = retry(baz, max_retry=2)  # direct call style
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            delay = 1.0
            for attempt in range(max_retry + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Retrying {f.__name__} due to error: {e}")
                    if attempt == max_retry:
                        # out of retries: re-raise last exception
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

        return wrapper

    # allow both @retry(...) and retry(func, ...)
    if callable(func):
        return decorator(func)
    return decorator


def write_redis_config(port, logfile, file_path="/opt/redis_config.conf"):
    """
    Write the redis config file.
    redis_config_path: the path to the redis config file.
    port: the port for Redis to listen on.
    logfile: the logfile for Redis.

    return the actual path of the redis config file.
    """
    config_content = f"""# Redis configuration file example for insecure connections

# Bind to all network interfaces (use with caution)
bind 0.0.0.0

# Set the port for Redis to listen on (default is {port})
port {port}

# Disable TLS by setting the tls-port to 0
tls-port 0

# Disable authentication by commenting out the requirepass directive
# requirepass yourpassword

# Other configuration settings can remain as default or be customized as needed
timeout 0
tcp-keepalive 300
protected-mode no
# enable-protected-configs yes
# enable-debug-command yes
# enable-module-command yes
daemonize yes
supervised no
loglevel notice
logfile {logfile}
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /opt
"""
    with open(file_path, "w") as file:
        file.write(config_content)
    return file_path


def do_once(func):
    """Decorator to make a function only call once."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = func.__name__ + str(args) + str(kwargs)
        if key not in do_once.dones:
            do_once.dones.add(key)
            return func(*args, **kwargs)

    return wrapper


do_once.dones = set()


def is_master_rank(parallel_dims, global_rank: int):
    return (not parallel_dims.pp_enabled and global_rank == 0) or (
        parallel_dims.pp_enabled
        and global_rank
        == (parallel_dims.world_size - parallel_dims.world_size / parallel_dims.pp)
    )


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)


def is_cuda_compatible(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


# From torchao.
def parse_version(version_string):
    # Extract just the X.Y.Z part from the version string
    match = re.match(r"(\d+\.\d+\.\d+)", version_string)
    if match:
        version = match.group(1)
        return [int(x) for x in version.split(".")]
    else:
        raise ValueError(f"Invalid version string format: {version_string}")


def compare_versions(v1, v2):
    v1_parts = parse_version(v1)
    v2_parts = parse_version(v2)
    return (v1_parts > v2_parts) - (v1_parts < v2_parts)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


def torch_version_at_least(min_version):
    return is_fbcode() or compare_versions(torch.__version__, min_version) >= 0


class TrieNode:
    __slots__ = ("children", "idxs")

    def __init__(self):
        self.children = {}
        self.idxs = []


def build_trie(seqs):
    root = TrieNode()
    for i, s in enumerate(seqs):
        node = root
        for x in s:
            node.idxs.append(i)
            node = node.children.setdefault(x, TrieNode())
        node.idxs.append(i)
    return root


def find_maximal_prefix_groups(
    seqs: List[List[int]],
    N: int,
):
    if N is None or N < 1:
        return {}
    root = build_trie(seqs)
    result = {}
    # We'll do a post-order traversal with an explicit stack
    # Each frame is (node, prefix, visited_flag)
    stack = [(root, (), False)]
    while stack:
        node, prefix, visited = stack.pop()
        if not visited:
            # push back as “to be processed after children”
            stack.append((node, prefix, True))
            # push children
            for x, child in node.children.items():
                stack.append((child, prefix + (x,), False))
        else:
            # now all children have been “visited” so we can decide if
            # this node is a maximal group
            if len(prefix) >= N and len(node.idxs) > 1:
                # check if any child also has ≥2 idxs
                has_deeper = any(
                    len(child.idxs) > 1 for child in node.children.values()
                )
                if not has_deeper:
                    result[prefix] = list(node.idxs)
    return result


def add_nan_checks(model):
    """
    Add nan checks to the model.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad or not param.is_leaf:
            continue

        # factory to capture the current name in a closure
        def make_hook(param_name):
            def hook(grad):
                origin_grad = grad
                if isinstance(grad, torch.distributed.tensor.DTensor):
                    grad = grad.to_local()
                if torch.isnan(grad).any():
                    msg = f"NaN detected in gradient of {param_name}"
                    raise RuntimeError(msg)
                return origin_grad  # must return the (possibly modified) grad

            return hook

        param.register_hook(make_hook(name))
        logger.info(f"Added nan check for {name}")


# Util func to create an asyncio task with proper cleanup
strong_refs = set()


def create_async_task(coro):
    task = asyncio.create_task(coro)
    strong_refs.add(task)
    task.add_done_callback(strong_refs.discard)
    return task


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 2048):
    """Memory-efficient entropy calculation with chunking."""
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i : i + chunk_size].float()
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(
            pd_chunk * logits_chunk, dim=-1
        )
        entropy[i : i + chunk_size] = entropy_chunk
    return entropy


def compute_logprobs(
    input_ids_batch: torch.Tensor,  # [batch_size, max_len]
    logprob_masks: torch.Tensor,  # [batch_size, max_len],
    logits: torch.Tensor,  # [batch_size, max_len, vocab_size] or [n_logprob_tokens, vocab_size] if is_full_logits is False
    is_full_logits: bool = False,
    label_packing_mask: Optional[torch.Tensor] = None,  # [batch_size, max_len]
    input_packing_mask: Optional[torch.Tensor] = None,  # [batch_size, max_len]
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the per-token log probabilities and advantages

    Args:
        input_ids_batch: the input_ids of the model [batch_size, max_len]
        logprob_masks: the logprob_masks of the model [batch_size, max_len]
        logits: the logits of the model [batch_size, max_len, vocab_size] or [n_logprob_tokens, vocab_size]
        is_full_logits: whether the logits are full logits or have been index-selected for memory efficiency
        label_packing_mask: the packing mask for the labels, if using packed sequences
        input_packing_mask: the packing mask for the inputs, if using packed sequences

    Returns:
        logps: the per-token log probabilities
        cu_seqlens: the cumulative sequence lengths of the logps
        metrics_dict: a dict of collected metrics, including entropy
    """
    # Shift token_ids
    if label_packing_mask is not None:
        assert (
            input_packing_mask is not None
        ), "input_packing_mask must be provided if label_packing_mask is used"
        shifted_input_ids = torch.zeros_like(input_ids_batch)
        shifted_input_ids[input_packing_mask] = input_ids_batch[label_packing_mask]
    else:
        shifted_input_ids = torch.empty_like(input_ids_batch)
        shifted_input_ids[:, :-1] = input_ids_batch[:, 1:]
        shifted_input_ids[:, -1] = 0

    if is_full_logits:
        assert (
            logits.shape[:2] == shifted_input_ids.shape[:2]
        ), f"Logits shape {logits.shape} does not match input_ids shape {shifted_input_ids.shape}"
        effective_logits = logits[logprob_masks]
    else:
        effective_logits = logits
    bsz, _ = input_ids_batch.shape

    effective_input_ids = shifted_input_ids[logprob_masks]  # [n_logprob_tokens,]
    # Compute sft cross entropy loss
    # sft_loss = torch.nn.functional.cross_entropy(
    #     effective_logits.view(-1, effective_logits.size(-1)),
    #     effective_input_ids.view(-1),
    #     reduction="mean",
    # )
    # print(f"[Policy] SFT loss: {sft_loss}, shape: {effective_input_ids.shape, effective_logits.shape}")

    metrics_dict = {}
    with torch.no_grad():
        # Compute entropy for logging with chunking to save memory
        entropy = entropy_from_logits_with_chunking(
            logits.view(-1, logits.size(-1))
        ).detach()
        metrics_dict["entropy"] = entropy.mean()
        metrics_dict["effective_entropy"] = (
            entropy[logprob_masks.view(-1)].mean()
            if is_full_logits
            else metrics_dict["entropy"]
        )

    masked_seqlens = logprob_masks.sum(dim=-1)  # [bsz,]
    cu_seqlens = torch.zeros(
        bsz + 1, dtype=torch.int32, device=logits.device
    )  # [bsz + 1,]
    cu_seqlens[1:] = torch.cumsum(masked_seqlens, dim=0)
    logps = selective_log_softmax(
        effective_logits, effective_input_ids
    )  # [n_logprob_tokens,]
    return logps, cu_seqlens, metrics_dict


def dynamic_import_module(path: str, attr: Optional[str] = None) -> Dict[str, Any]:
    """
    Dynamically import either:
        - a single .py file
        - a package directory (must contain __init__.py)
    and allow it to use relative imports internally.

    Args:
        path: the path to the module to import
        attr: the attribute to import from the module, can be recursive attribute like `model.attr_a.attr_b.attr_c....`

    Returns the imported module object.
    """
    if os.path.exists(path):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            # it's a package dir
            pkg_dir = path
            if not os.path.isfile(os.path.join(pkg_dir, "__init__.py")):
                raise ImportError(f"{pkg_dir!r} is not a package (no __init__.py)")
            module_name = os.path.basename(pkg_dir)
            parent_dir = os.path.dirname(pkg_dir)
        else:
            # it's a single .py file
            if not path.lower().endswith(".py"):
                raise ImportError(
                    f"{path!r} is neither a .py file nor a package directory"
                )
            parent_dir, filename = os.path.split(path)
            module_name = os.path.splitext(filename)[0]
        # Ensure the parent directory is on sys.path
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
    else:
        # Direct python module
        module_name = path

    # Now import by name – normal import machinery applies
    module = importlib.import_module(module_name)

    if attr:
        obj = module
        for attr_part in attr.split("."):
            if not hasattr(obj, attr_part):
                raise ImportError(f"Attribute {attr} not found in {path}")
            obj = getattr(obj, attr_part)
        return obj
    return module


class RollingDict(OrderedDict):
    def __init__(self, maxlen=20):
        super().__init__()
        self.maxlen = maxlen

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        if len(self) > self.maxlen:
            self.popitem(last=False)


def sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    else:
        return obj


def safe_deep_getattr(obj, attr_path, default=None):
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default


def load_model_class_by_config(hf_config):
    architectures = hf_config.architectures
    assert (
        len(architectures) == 1
    ), f"zero or multiple architectures in config.json is not supported, {architectures=}"

    class_name = architectures[0]
    model_class = None
    try:
        model_class = getattr(importlib.import_module("transformers"), class_name)
    except Exception as e:
        logger.error(f"Can not load {class_name} from transformers")
        raise e
    return model_class


def reverse_hf_checkpoint_mapping(mapping):
    reverse_map = {}
    for k, v in mapping.items():
        reverse_map[r"^" + re.escape(v)] = k.lstrip("^")
    return reverse_map


def decode_vision_info(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = prompt.copy()
        if "multi_modal_data" in new_prompt:  # Skip if text-only prompt
            multi_modal_data = new_prompt.pop("multi_modal_data")
            if "image" in multi_modal_data:
                img_obj = multi_modal_data["image"]
                assert isinstance(
                    img_obj, list
                ), f"image should be a list, but got {type(img_obj)}"
                img_type = type(img_obj[0])
                if img_type == str:
                    for i, img_b64 in enumerate(img_obj):
                        img_bytes = base64.b64decode(img_b64)
                        img_buffer = io.BytesIO(img_bytes)
                        image = Image.open(img_buffer)
                        multi_modal_data["image"][i] = image
                elif img_type == torch.Tensor:
                    pass
                else:
                    raise ValueError(f"Unsupported image type: {img_type}")
            if "video" in multi_modal_data:
                video_obj = multi_modal_data["video"]
                assert isinstance(
                    video_obj, list
                ), f"video should be a list, but got {type(video_obj)}"
                video_type = type(video_obj[0])
                # No need to decode video tensor
                if video_type == torch.Tensor:
                    pass
                else:
                    raise ValueError(f"Unsupported video type: {video_type}")

            new_prompt["multi_modal_data"] = multi_modal_data
        new_prompts.append(new_prompt)
    return new_prompts


def rank0_print(msg, *args, **kwargs):
    if os.environ.get("RANK", "0") == "0":
        logger.info(msg, *args, **kwargs)


def replace_with_liger_equivalents(root: torch.nn.Module) -> None:
    # Walk children first so we can safely replace in-place.
    for name, child in list(root.named_children()):
        replace_with_liger_equivalents(child)

        lig = getattr(child, "liger_equivalent", None)
        if lig is None:
            continue

        # Call the factory/method if it's callable; otherwise assume it's already a module.
        new_child = lig() if callable(lig) else lig

        # Keep training/eval state.
        try:
            new_child.train(child.training)
        except Exception:
            pass

        # Try to place on the same device/dtype as the original.
        ref = next(child.parameters(recurse=True), None)
        if ref is None:
            ref = next(child.buffers(recurse=True), None)
        if ref is not None:
            try:
                if ref.is_floating_point():
                    new_child.to(device=ref.device, dtype=ref.dtype)
                else:
                    new_child.to(device=ref.device)
            except Exception:
                pass
        rank0_print(f"Replaced {name} with liger equivalent")
        # Swap it in.
        setattr(root, name, new_child)
