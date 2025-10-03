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

#!/usr/bin/env python3

import socket
import subprocess
import sys
import logging
import time
import os
import shutil
import re
import argparse
from argparse import REMAINDER
from typing import List, Dict, Optional, Any, Callable
import toml
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cosmos")


# ---------------------------------------------------------------------------
# Queue priority helper
# ---------------------------------------------------------------------------
# Map numeric queue-priority (1-9) to the corresponding Lepton priority_class
# string expected by the backend API.
#
# 1-3  → low-1000 / 2000 / 3000
# 4-6  → mid-4000 / 5000 / 6000
# 7-9  → high-7000 / 8000 / 9000
#
# Note: keep in sync with lepton-cli definitions.
NUM_PRIORITY_MAPPING = {
    1: "low-1000",
    2: "low-2000",
    3: "low-3000",
    4: "mid-4000",
    5: "mid-5000",
    6: "mid-6000",
    7: "high-7000",
    8: "high-8000",
    9: "high-9000",
}


def wait_for_url_ready(url: str, process: Optional[subprocess.Popen] = None):
    """
    Wait for a URL to be ready by sending a GET request.

    Args:
        url: The URL to check

    Returns:
        None
    """
    while True:
        # create TCP socket
        try:
            if process is not None:
                if process.poll() is not None:
                    if process.returncode != 0:
                        logger.error(
                            f"Process {process.pid} exited with code {process.returncode}. Exiting."
                        )
                        sys.exit(1)
                    else:
                        logger.error(
                            f"Process {process.pid} exited as soon as launched. Exiting."
                        )
                        sys.exit(1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            host, port = url.split(":")
            sock.connect((host, int(port)))
            sock.close()
            break
        except socket.error:
            # If the connection fails, wait and retry
            time.sleep(1)


def read_config(config_file: str) -> Dict[str, Any]:
    """
    Read configuration from a TOML file.

    Args:
        config_file: Path to the TOML configuration file

    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_file, "r") as f:
            config = toml.load(f)
        return config
    except Exception as e:
        logger.error(f"Error reading config file {config_file}: {e}")
        sys.exit(1)


def get_available_gpus() -> List[str]:
    """
    Detect available GPUs using nvidia-smi and return their IDs.

    Returns:
        List of GPU IDs as strings
    """
    try:
        cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        cvd = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cvd is not None:
            # Add the GPU IDs to the command
            cmd += ["--id=" + cvd]
        # Run nvidia-smi to get GPU information
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the output to get GPU IDs
        gpu_ids = [line.strip() for line in result.stdout.splitlines()]

        if not gpu_ids:
            logger.error("Warning: No GPUs detected")
            return []

        logger.info(f"Detected {len(gpu_ids)} GPUs: {', '.join(gpu_ids)}")
        return gpu_ids

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvidia-smi: {e}")
        return []
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return []


def launch_processes(
    commands: List[str],
    gpu_devices: Optional[List[str]],
    control_urls: Optional[List[str]],
    output_files: Optional[List[str]],
    extra_env: Optional[Dict[str, str]] = None,
) -> List[subprocess.Popen]:
    """
    Launch multiple subprocesses and return their process objects.

    Args:
        commands: List of command strings to execute
        gpu_devices: List of GPU device IDs to assign to each process (e.g., ["0", "1", "2"])
        control_urls: List of controller URLs to assign to each process (e.g., ["localhost:8000"])
        output_files: List of output files to redirect process output to (e.g., ["output1.log", "output2.log"])

    Returns:
        List of Popen objects for the launched processes
    """
    processes = []

    if gpu_devices is None:
        gpu_devices = [None] * len(commands)
    elif len(gpu_devices) != len(commands):
        raise ValueError("Number of GPU devices must match number of commands")

    for cmd, gpu_id, url, ofile in zip(
        commands, gpu_devices, control_urls, output_files
    ):
        try:
            # Prepare environment variables
            env = dict(os.environ)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            if url is not None:
                env["COSMOS_CONTROLLER_HOST"] = url
            if extra_env is not None:
                env.update(extra_env)
            if ofile is not None:
                f = open(ofile, "wb")
                cout = f
                cerr = f
            else:
                cout = sys.stdout
                cerr = sys.stderr

            # Launch process and capture output
            logger.info(f"Launching process with command: {cmd}")
            process = subprocess.Popen(
                cmd, shell=True, stdout=cout, stderr=cerr, env=env
            )
            processes.append(process)
            if ofile is not None:
                f.close()
        except Exception as e:
            logger.error(f"Error launching process for command '{cmd}': {e}")

    return processes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multiple processes with GPU assignments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML configuration file, which specifies the detailed configuration for the whole training process including algorithm, model, data, parallelism, etc.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of the controller for the policy and rollout replicas to connect to, consisting of IP and port in the format ip:port. If not provided, the controller will be launched on the local machine. If provided and the IP is the local IP, the controller will be launched on the local machine. If provided and the IP is not the local IP, the controller will be launched on the remote machine.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the controller to connect to, default is 8000. This is only used when --url is not provided to launch the controller on the local machine.",
    )
    parser.add_argument(
        "--policy",
        type=int,
        default=None,
        help="Total number of policy replicas to launch in the whole system. If not provided, the number of policy replicas will be obtained from TOML configuration file.",
    )
    parser.add_argument(
        "--rollout",
        type=int,
        default=None,
        help="Total number of rollout replicas to launch in the whole system. If not provided, the number of rollout replicas will be obtained from TOML configuration file.",
    )
    parser.add_argument(
        "--p2r-ratio",
        type=str,
        default=None,
        help="Ratio of policy replicas to rollout replicas. This is used to determine the number of rollout replicas and the number of policy replicas based on the number of workers.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save logs. If not provided, logs will be printed to stdout.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers to use for the job, default is 1. This is used when multi-node training are used for the job.",
    )
    parser.add_argument(
        "--worker-idx",
        type=int,
        default=0,
        help="Worker index for local execution. In Lepton mode, this is ignored as worker indices are automatically assigned by Lepton.",
    )

    parser.add_argument(
        "--node-ip-list",
        type=str,
        default=None,
        help="list of ips for all the workers, separated by ';'. This is used when multi-node training are used for one replica.",
    )

    parser.add_argument(
        "--rdzv-port",
        type=int,
        default=29345,
        help="Rendezvous endpoint port for the job, default is 29345. This is used when multi-node training are used for one replica.",
    )

    parser.add_argument(
        "--lepton-mode",
        action="store_true",
        default=False,
        help="Enable Lepton mode for remote execution",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable log level to debug",
    )

    # Lepton specific options
    lepton_group = parser.add_argument_group("Lepton mode options")
    lepton_group.add_argument("--lepton-job-name", "-n", type=str, help="Job name")
    lepton_group.add_argument(
        "--lepton-container-image", type=str, help="Container image for the job"
    )
    lepton_group.add_argument(
        "--lepton-container-port",
        type=str,
        help="Ports to expose for the job, in the format portnumber[:protocol]",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-resource-shape", type=str, help="Resource shape for the pod"
    )
    lepton_group.add_argument(
        "--lepton-node-group",
        "-ng",
        type=str,
        help="Node group for the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-max-failure-retry",
        type=int,
        help="Maximum number of failures to retry per worker",
    )
    lepton_group.add_argument(
        "--lepton-max-job-failure-retry",
        type=int,
        help="Maximum number of failures to retry per whole job",
    )
    lepton_group.add_argument(
        "--lepton-env",
        "-e",
        type=str,
        help="Environment variables to pass to the job, in the format `NAME=VALUE`",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-secret",
        "-s",
        type=str,
        help="Secrets to pass to the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-mount",
        type=str,
        help="Persistent storage to be mounted to the job",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-image-pull-secrets",
        type=str,
        help="Secrets to use for pulling images",
        action="append",
    )
    lepton_group.add_argument(
        "--lepton-intra-job-communication",
        type=bool,
        help="Enable intra-job communication",
    )
    lepton_group.add_argument(
        "--lepton-privileged",
        action="store_true",
        help="Run the job in privileged mode",
    )
    lepton_group.add_argument(
        "--lepton-ttl-seconds-after-finished",
        type=int,
        help="TTL for finished jobs in seconds",
        default=259200,
    )
    lepton_group.add_argument(
        "--lepton-log-collection",
        "-lg",
        type=bool,
        help="Enable or disable log collection",
    )
    lepton_group.add_argument(
        "--lepton-node-id", "-ni", type=str, help="Node for the job", action="append"
    )
    lepton_group.add_argument(
        "--lepton-queue-priority",
        "-qp",
        type=int,
        choices=list(NUM_PRIORITY_MAPPING.keys()),
        help=(
            "Queue priority for dedicated node groups. Provide a number 1-9 which"
            " will be mapped to priority classes low-1000 … high-9000."
        ),
    )
    # Whether the job can be preempted by higher-priority jobs (only valid for
    # dedicated node groups). Tri-state: flag present → True; absent → None.
    lepton_group.add_argument(
        "--lepton-can-be-preempted",
        "-cbp",
        action="store_true",
        default=None,
        help=(
            "Allow this job to be preempted by higher priority jobs (only for"
            " dedicated node groups)."
        ),
    )

    # Whether the job itself is allowed to preempt lower-priority jobs.
    lepton_group.add_argument(
        "--lepton-can-preempt",
        "-cp",
        action="store_true",
        default=None,
        help=(
            "Allow this job to preempt lower priority jobs (only for dedicated"
            " node groups)."
        ),
    )
    lepton_group.add_argument(
        "--lepton-visibility", type=str, help="Visibility of the job (public/private)"
    )
    lepton_group.add_argument(
        "--lepton-shared-memory-size", type=int, help="Shared memory size in MiB"
    )
    lepton_group.add_argument(
        "--lepton-with-reservation",
        type=str,
        help="Reservation ID for dedicated node groups",
    )

    # Positional arguments

    parser.add_argument(
        "script",
        nargs="?",  # “?” means 0 or 1 occurrences
        default=None,
        help="A user script which can be provided for custom dataset, reward functions, and model registration.",
    )

    parser.add_argument("script_args", nargs=REMAINDER)

    args = parser.parse_args()

    # Validate Lepton mode arguments
    if args.lepton_mode:
        required_args = [("lepton_job_name", "--lepton-job-name")]

        for arg_name, arg_flag in required_args:
            if not getattr(args, arg_name):
                parser.error(f"{arg_flag} is required when --lepton-mode is enabled")

    return args, parser


def get_local_ip():
    """
    Get the local IP address of the machine.

    Returns:
        Local IP address as a string
    """
    try:
        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return [local_ip, hostname]
    except Exception as e:
        logger.error(f"Error getting local IP address: {e}")
        return None


def replica_placement(
    available_gpus: List[int],
    n_policy: int,
    n_rollouts: int,
    min_n_gpus_policy: int,
    min_n_gpus_rollout: int,
    replica_script: str,
    control_url: str,
    output_dir: Optional[str],
    get_worker_ip: Optional[Callable] = None,
    rdzv_port: Optional[int] = None,
    script: Optional[str] = None,
    backend: str = "vllm",
    config_path: Optional[str] = None,
    script_args: Optional[List[Any]] = None,
) -> List[List[str]]:
    commands = []
    gpu_devices = []
    control_urls = []
    output_files = []
    assert len(available_gpus) in [
        1,
        2,
        4,
        8,
    ], "Number of GPUs per worker must be 1, 2, 4, or 8"
    # Prepare the command to launch the controller for all workers
    global_available_gpus = [available_gpus]
    # Create commands for policy and rollout replicas
    gpu_idx = 0
    global_worker_idx = 0
    global_launch_settings = []
    # Assign launch settings for each worker
    for i in range(n_policy):
        if min_n_gpus_policy > len(global_available_gpus[global_worker_idx]):
            assert (
                min_n_gpus_policy % len(global_available_gpus[global_worker_idx]) == 0
            ), f"min_n_gpus_policy {min_n_gpus_policy} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus_policy // len(
                global_available_gpus[global_worker_idx]
            )
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices.append(
                    ",".join([str(g) for g in global_available_gpus[global_worker_idx]])
                )
                commands.append(
                    f"{replica_script} --type policy --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed} --backend {backend} --config {config_path}"
                )
                if script is not None:
                    commands[-1] += f" --script {script}"
                if node_in_replica == 0:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx)
                else:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"

                if script_args is not None:
                    commands[-1] += f" {' '.join(script_args)}"

                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"policy_{i}.log")
                    if output_dir is not None
                    else None
                )
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)
        else:
            if gpu_idx + min_n_gpus_policy > len(
                global_available_gpus[global_worker_idx]
            ):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)

            gpu_devices.append(
                ",".join(
                    [
                        str(g)
                        for g in global_available_gpus[global_worker_idx][
                            gpu_idx : gpu_idx + min_n_gpus_policy
                        ]
                    ]
                )
            )
            commands.append(
                f"{replica_script} --type policy --ngpus {min_n_gpus_policy} --backend {backend} --config {config_path}"
            )
            if script is not None:
                commands[-1] += f" --script {script}"
            if script_args is not None:
                commands[-1] += f" {' '.join(script_args)}"
            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"policy_{i}.log")
                if output_dir is not None
                else None
            )
            gpu_idx += min_n_gpus_policy

    if min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
        # If the number of GPUs needed for rollout is more than available GPUs, we need to allocate a new worker
        if gpu_idx > 0:
            global_launch_settings.append(
                [commands, gpu_devices, control_urls, output_files]
            )
            commands = []
            gpu_devices = []
            control_urls = []
            output_files = []
            gpu_idx = 0
            global_worker_idx += 1
            global_available_gpus.append(available_gpus)

    for i in range(n_rollouts):
        if min_n_gpus_rollout > len(global_available_gpus[global_worker_idx]):
            assert (
                min_n_gpus_rollout % len(global_available_gpus[global_worker_idx]) == 0
            ), f"min_n_gpus_rollout {min_n_gpus_rollout} is not divisible by {len(global_available_gpus[global_worker_idx])}"
            nodes_needed = min_n_gpus_rollout // len(
                global_available_gpus[global_worker_idx]
            )
            rdzv_ip = "localhost"
            for node_in_replica in range(nodes_needed):
                gpu_devices.append(
                    ",".join([str(g) for g in global_available_gpus[global_worker_idx]])
                )
                commands.append(
                    f"{replica_script} --type rollout --ngpus {len(global_available_gpus[global_worker_idx])} --nnodes {nodes_needed} --backend {backend} --config {config_path}"
                )
                if script is not None:
                    commands[-1] += f" --script {script}"
                if node_in_replica == 0:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"
                    if get_worker_ip is not None:
                        rdzv_ip = get_worker_ip(global_worker_idx)
                else:
                    commands[-1] += f" --rdzv-endpoint {rdzv_ip}:{rdzv_port}"

                if script_args is not None:
                    commands[-1] += f" {' '.join(script_args)}"

                control_urls.append(control_url)
                output_files.append(
                    os.path.join(output_dir, f"rollout_{i}.log")
                    if output_dir is not None
                    else None
                )
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)
        else:
            if gpu_idx + min_n_gpus_rollout > len(
                global_available_gpus[global_worker_idx]
            ):
                global_launch_settings.append(
                    [commands, gpu_devices, control_urls, output_files]
                )
                commands = []
                gpu_devices = []
                control_urls = []
                output_files = []
                gpu_idx = 0
                global_worker_idx += 1
                global_available_gpus.append(available_gpus)

            gpu_devices.append(
                ",".join(
                    [
                        str(g)
                        for g in available_gpus[gpu_idx : gpu_idx + min_n_gpus_rollout]
                    ]
                )
            )
            commands.append(
                f"{replica_script} --type rollout --ngpus {min_n_gpus_rollout} --backend {backend} --config {config_path}"
            )
            if script is not None:
                commands[-1] += f" --script {script}"

            if script_args is not None:
                commands[-1] += f" {' '.join(script_args)}"

            control_urls.append(control_url)
            output_files.append(
                os.path.join(output_dir, f"rollout_{i}.log")
                if output_dir is not None
                else None
            )
            gpu_idx += min_n_gpus_rollout

    if len(commands) > 0:
        global_launch_settings.append(
            [commands, gpu_devices, control_urls, output_files]
        )
    return global_launch_settings


def get_hostname_from_host(ip):
    try:
        # Run 'host' command
        result = subprocess.run(
            ["host", ip], capture_output=True, text=True, check=True
        )
        # Parse output
        output = result.stdout.strip()
        if "has address" in output:
            # Extract part after "has address "
            hostname = output.rsplit("has address ", 2)[-1].strip()
            return hostname
        else:
            return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def main():
    args, parser = parse_args()
    if args.debug:
        os.environ["COSMOS_LOG_LEVEL"] = "DEBUG"

    # Check if the config file is provided
    cosmos_config = read_config(args.config)
    if args.script is not None and args.script.endswith(".py"):
        # If the script is a Python file, we need to make sure it is absolute path
        # so that it can be found by the launched processes
        script = os.path.abspath(args.script)
    else:
        script = args.script if args.script is not None else None

    # Get the number of GPUs required for policy and rollout
    # and the number of replicas for each
    policy_parallelism = cosmos_config.get("policy", {}).get("parallelism", {})
    rollout_parallelism = cosmos_config.get("rollout", {}).get("parallelism", {})
    # Calculate the minimum number of GPUs required for policy and rollout
    # based on the parallelism settings in the configuration
    # Treat dp_shard_size as 1 if it is not set
    min_n_gpus_policy = (
        policy_parallelism.get("tp_size", 1)
        * policy_parallelism.get("dp_replicate_size", 1)
        * policy_parallelism.get("pp_size", 1)
        * policy_parallelism.get("cp_size", 1)
    )
    min_n_gpus_rollout = (
        rollout_parallelism.get("tp_size", 1)
        * rollout_parallelism.get("dp_replicate_size", 1)
        * rollout_parallelism.get("pp_size", 1)
        * rollout_parallelism.get("cp_size", 1)
    )
    if policy_parallelism.get("dp_shard_size", 1) >= 1:
        min_n_gpus_policy = min_n_gpus_policy * policy_parallelism.get(
            "dp_shard_size", 1
        )
    if rollout_parallelism.get("dp_shard_size", 1) >= 1:
        min_n_gpus_rollout = min_n_gpus_rollout * rollout_parallelism.get(
            "dp_shard_size", 1
        )
    backend = cosmos_config.get("rollout", {}).get("backend", "vllm")
    logger.info(f"Using rollout backend: {backend}")

    if args.p2r_ratio is not None:
        assert (
            args.num_workers is not None
        ), "When using --p2r-ratio, --num-workers must be specified"
        p2r_ratio = args.p2r_ratio.split(":")
        assert (
            len(p2r_ratio) == 2
        ), "Invalid --p2r-ratio format. Use 'policy:rollout' format."
        p_ratio = int(p2r_ratio[0])
        r_ratio = int(p2r_ratio[1])

        if args.lepton_mode:
            match = re.search(r"(8|4|2)x", args.lepton_resource_shape)
            if match:
                num_gpus_per_node = int(match.group(1))
            else:
                num_gpus_per_node = 1
        else:
            num_gpus_per_node = len(get_available_gpus())

        num_per_ratio = (
            args.num_workers
            * num_gpus_per_node
            / (p_ratio * min_n_gpus_policy + r_ratio * min_n_gpus_rollout)
        )
        args.policy = int(num_per_ratio * p_ratio)
        args.rollout = int(num_per_ratio * r_ratio)
        assert args.policy >= 1, "Number of policy replicas must be at least 1"
        assert (
            args.policy * min_n_gpus_policy + args.rollout * min_n_gpus_rollout
            <= args.num_workers * num_gpus_per_node
        )

    if args.policy is None:
        n_policy = policy_parallelism.get("n_init_replicas", 1)
    else:
        n_policy = args.policy
    if args.rollout is None:
        n_rollouts = rollout_parallelism.get("n_init_replicas", 1)
    else:
        n_rollouts = args.rollout

    # If the training type is SFT, set n_rollouts to 0
    if (
        cosmos_config.get("train", {}).get("train_policy", {}).get("type", "grpo")
        == "sft"
    ):
        n_rollouts = 0
        if n_policy > 1:
            logger.warning(
                "Warning: n_init_replicas for rollout is set to 0 for SFT training, but n_init_replicas for policy is more than 1."
            )
            pre_dp_replicate_size = policy_parallelism.get("dp_replicate_size", 1)
            cosmos_config["policy"]["parallelism"]["dp_replicate_size"] = (
                pre_dp_replicate_size * n_policy
            )
            logger.info(
                f"[Config ]SFT type job does not support n_init_replicas > 1, automatically set n_init_replicas from {n_policy} to 1 and scale up dp_replicate_size from {pre_dp_replicate_size} to {cosmos_config['policy']['parallelism']['dp_replicate_size']}."
            )
            min_n_gpus_policy = min_n_gpus_policy * n_policy
            n_policy = 1

    # Handle Lepton mode
    if args.lepton_mode:
        from leptonai.api.v2.client import APIClient
        from leptonai.config import BASE_IMAGE, VALID_SHAPES
        from leptonai.api.v1.types.job import (
            LeptonJob,
            LeptonJobUserSpec,
            LeptonResourceAffinity,
            ReservationConfig,
        )
        from leptonai.api.v1.types.deployment import (
            LeptonLog,
            QueueConfig,
        )
        from leptonai.api.v1.types.common import Metadata, LeptonVisibility
        from leptonai.api.v1.photon import (
            make_env_vars_from_strings,
            make_mounts_from_strings,
        )
        from leptonai.cli.util import (
            _get_valid_nodegroup_ids,
            _get_valid_node_ids,
            make_container_port_from_string,
        )

        # Initialize Lepton client
        client = APIClient()

        # Create job specification
        job_spec = LeptonJobUserSpec()

        # Construct the original launch_processes command
        # Update policy and rollout numbers in the lepton config
        if "policy" in cosmos_config and "parallelism" in cosmos_config["policy"]:
            cosmos_config["policy"]["parallelism"]["n_init_replicas"] = n_policy
        if "rollout" in cosmos_config and "parallelism" in cosmos_config["rollout"]:
            cosmos_config["rollout"]["parallelism"]["n_init_replicas"] = n_rollouts
        config_content = toml.dumps(cosmos_config)
        launch_cmd = f"""\
cat >config.toml <<EOF
{config_content}
EOF

cosmos-rl --config config.toml"""

        # Get all non-Lepton arguments
        non_lepton_args = []
        for action in parser._actions:
            if hasattr(action, "option_strings") and action.option_strings:
                # Skip help action, lepton related arguments, and worker-idx
                if (
                    action.dest == "help"
                    or any(
                        opt.startswith("--lepton-") or opt == "--lepton-mode"
                        for opt in action.option_strings
                    )
                    or action.dest == "worker_idx"
                    or action.dest == "config"
                ):  # skip worker-idx
                    continue

                value = getattr(args, action.dest)
                if value is not None:
                    if isinstance(value, bool):
                        if value:
                            non_lepton_args.append(action.option_strings[0])
                    else:
                        non_lepton_args.append(f"{action.option_strings[0]} {value}")

        # Add all non-Lepton arguments to the command
        launch_cmd += " " + " ".join(non_lepton_args)
        if script is not None:
            launch_cmd += f" {script}"

        # Handle node groups, queue priority and preemption flags
        if (
            args.lepton_node_group
            or args.lepton_queue_priority is not None
            or args.lepton_can_be_preempted is not None
            or args.lepton_can_preempt is not None
        ):
            if (
                args.lepton_queue_priority is not None
                or args.lepton_can_be_preempted is not None
                or args.lepton_can_preempt is not None
            ) and not args.lepton_node_group:
                logger.error(
                    "Error: Queue priority is only available for dedicated node groups"
                )
                logger.error(
                    "Please use --lepton-queue-priority with --lepton-node-group"
                )
                sys.exit(1)

            node_group_ids = _get_valid_nodegroup_ids(
                args.lepton_node_group,
                need_queue_priority=(
                    args.lepton_queue_priority is not None
                    or args.lepton_can_be_preempted is not None
                    or args.lepton_can_preempt is not None
                ),
            )
            valid_node_ids = (
                _get_valid_node_ids(node_group_ids, args.lepton_node_id)
                if args.lepton_node_id
                else None
            )

            job_spec.affinity = LeptonResourceAffinity(
                allowed_dedicated_node_groups=node_group_ids,
                allowed_nodes_in_node_group=valid_node_ids,
            )

            if (
                args.lepton_queue_priority is not None
                or args.lepton_can_be_preempted is not None
                or args.lepton_can_preempt is not None
            ):
                # Ensure queue_config exists
                if job_spec.queue_config is None:
                    job_spec.queue_config = QueueConfig()

                priority_class = None
                if args.lepton_queue_priority is not None:
                    # Convert numeric priority to the Lepton priority_class string.
                    priority_class = NUM_PRIORITY_MAPPING[args.lepton_queue_priority]

                job_spec.queue_config.priority_class = priority_class or "mid-4000"

                if args.lepton_can_be_preempted is not None:
                    job_spec.queue_config.can_be_preempted = bool(
                        args.lepton_can_be_preempted
                    )

                if args.lepton_can_preempt is not None:
                    job_spec.queue_config.can_preempt = bool(args.lepton_can_preempt)

        # Set resource shape
        if args.lepton_resource_shape:
            job_spec.resource_shape = args.lepton_resource_shape
        else:
            available_types = "\n      ".join(VALID_SHAPES)
            logger.error(
                "Error: Missing option '--lepton-resource-shape'.\n"
                f"Available types are:\n      {available_types}.\n"
            )
            sys.exit(1)

        # Only for calculating the number of nodes needed.
        # Use this to replace the args.num_workers to calculate the number of nodes needed rather than specifying it using --num-workers
        # example

        match = re.search(r"(8|4|2)x", args.lepton_resource_shape)
        if match:
            num_gpus_per_node = int(match.group(1))
        else:
            num_gpus_per_node = 1
        global_launch_settings = replica_placement(
            list(range(num_gpus_per_node)),
            n_policy,
            n_rollouts,
            min_n_gpus_policy,
            min_n_gpus_rollout,
            replica_script="",
            control_url="",
            output_dir=None,
            script=script,
            backend=backend,
        )
        if args.num_workers is not None:
            assert args.num_workers >= len(global_launch_settings)
        num_workers = len(global_launch_settings)
        logger.info(f"Number of workers required: {num_workers}")

        # Handle workers and communication
        if num_workers > 0:
            job_spec.completions = num_workers
            job_spec.parallelism = num_workers
            job_spec.intra_job_communication = True
        elif args.lepton_intra_job_communication is not None:
            job_spec.intra_job_communication = args.lepton_intra_job_communication

        # Set failure retry settings
        if args.lepton_max_failure_retry:
            job_spec.max_failure_retry = args.lepton_max_failure_retry
        if args.lepton_max_job_failure_retry:
            job_spec.max_job_failure_retry = args.lepton_max_job_failure_retry

        # Handle command
        job_spec.container.command = ["/bin/bash", "-c", launch_cmd]

        # Set container image
        if args.lepton_container_image:
            job_spec.container.image = args.lepton_container_image
        else:
            job_spec.container.image = BASE_IMAGE

        # Handle ports
        if args.lepton_container_port:
            job_spec.container.ports = [
                make_container_port_from_string(p) for p in args.lepton_container_port
            ]

        # Handle environment variables and secrets
        if args.lepton_env or args.lepton_secret:
            job_spec.envs = make_env_vars_from_strings(
                args.lepton_env, args.lepton_secret
            )

        # Handle mounts
        if args.lepton_mount:
            job_spec.mounts = make_mounts_from_strings(args.lepton_mount)

        # Set other configurations
        if args.lepton_image_pull_secrets:
            job_spec.image_pull_secrets = args.lepton_image_pull_secrets
        if args.lepton_privileged:
            job_spec.privileged = args.lepton_privileged
        if args.lepton_ttl_seconds_after_finished:
            job_spec.ttl_seconds_after_finished = args.lepton_ttl_seconds_after_finished
        if args.lepton_log_collection is not None:
            job_spec.log = LeptonLog(enable_collection=args.lepton_log_collection)
        if args.lepton_shared_memory_size is not None:
            job_spec.shared_memory_size = args.lepton_shared_memory_size

        # Handle reservation
        if args.lepton_with_reservation:
            if not args.lepton_node_group:
                logger.error(
                    "Error: --lepton-with-reservation is only supported for dedicated node groups"
                )
                sys.exit(1)
            job_spec.reservation_config = ReservationConfig(
                reservation_id=args.lepton_with_reservation
            )

        # Create job
        job = LeptonJob(
            spec=job_spec,
            metadata=Metadata(
                id=args.lepton_job_name,
                visibility=LeptonVisibility(args.lepton_visibility)
                if args.lepton_visibility
                else None,
            ),
        )

        # Create the job
        created_job = client.job.create(job)
        new_job_id = created_job.metadata.id_
        logger.info("🎉 Job Created Successfully!")
        logger.info(f"Name: {args.lepton_job_name}")
        logger.info(f"ID: {new_job_id}")

        return

    import cosmos_rl.utils.util as util

    logger.info(
        f"Number of policy replicas: {n_policy} with {min_n_gpus_policy} gpus each"
    )
    logger.info(
        f"Number of rollout replicas: {n_rollouts} with {min_n_gpus_rollout} gpus each"
    )

    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available. Please check your GPU configuration.")

    # List of bash scripts to run (these should exist in the same directory)
    script_names = ["launch_controller.sh", "launch_replica.sh"]

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Verify scripts exist and are executable
    for script_name in script_names:
        script_path = os.path.join(script_dir, script_name)
        if not os.path.exists(script_path):
            logger.error(f"Error: Script {script_path} does not exist")
            sys.exit(1)
        if not os.access(script_path, os.X_OK):
            logger.error(f"Error: Script {script_path} is not executable")
            sys.exit(1)

    controller_script = os.path.join(script_dir, "launch_controller.sh")
    replica_script = os.path.join(script_dir, "launch_replica.sh")

    # Create commands for controller
    if args.log_dir is not None:
        output_dir = args.log_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(output_dir, f"logs_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = None

    def resolve_host(host):
        try:
            result = subprocess.run(
                ["getent", "hosts", "--", host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0:
                if len(result.stdout.strip().split()) > 0:
                    return result.stdout.strip().split()[0]
                else:
                    return None
            else:
                raise RuntimeError(f"Resolution failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise TimeoutError("DNS resolution timed out")

    def resolve_host_blocking(hostname):
        try:
            while True:
                new_hostname = resolve_host(hostname)
                if new_hostname is not None:
                    hostname = new_hostname
                    logger.info(f"Resolved hostname: {hostname}")
                    break
                time.sleep(1)
        except Exception:
            pass
        return hostname

    if (
        "LEPTON_JOB_WORKER_INDEX" in os.environ
        and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) >= 0
    ):
        cur_work_idx = int(os.environ.get("LEPTON_JOB_WORKER_INDEX"))
    else:
        cur_work_idx = args.worker_idx

    control_url = None
    if args.url is not None:
        ip, port = args.url.split(":")
        if ip in get_local_ip():
            # If the IP is the local IP, launch the controller on the local machine
            port = util.find_available_port(int(port))
            logger.info(f"Using local IP: {ip} so launching controller on port {port}")
        else:
            control_url = args.url
    else:
        if (
            "LEPTON_JOB_WORKER_INDEX" in os.environ
            and int(os.environ.get("LEPTON_JOB_WORKER_INDEX")) != 0
        ):
            # For non-primary workers, connect to the primary worker (index 0) using its hostname
            prefix = os.environ.get(
                "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
            )
            subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
            primary_hostname = f"{prefix}-0.{subdomain}"
            primary_hostname = resolve_host_blocking(primary_hostname)
            control_url = f"{primary_hostname}:{args.port}"
        elif "LEPTON_JOB_WORKER_INDEX" in os.environ:
            # If we're in a Lepton job prime node, check if the port is available
            if not util.is_port_free(args.port):
                raise RuntimeError(f"Port {args.port} is not available")
            else:
                port = args.port
        else:
            port = util.find_available_port(args.port)

    if control_url is None:
        logger.info(f"Controller will be launched locally on port {port}.")
    else:
        logger.info(
            f"Controller will be launched on another node. This node will connect to {control_url} for control."
        )

    controller_cmd = None
    tmpfile_toml = None
    if n_policy > 0 or n_rollouts > 0:
        # Do not update the config if no replicas are needed which means launch controller only.
        if "policy" in cosmos_config and "parallelism" in cosmos_config["policy"]:
            cosmos_config["policy"]["parallelism"]["n_init_replicas"] = n_policy
        if "rollout" in cosmos_config and "parallelism" in cosmos_config["rollout"]:
            # Only available for RL.
            cosmos_config["rollout"]["parallelism"]["n_init_replicas"] = n_rollouts
    # Create a temporary file and write to it
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".toml", delete=False
    ) as tmpfile:
        toml.dump(cosmos_config, tmpfile)
        tmpfile_toml = tmpfile.name

    if control_url is None:
        logger.info(f"Temporary configuration file created at {tmpfile_toml}")
        controller_cmd = f"{controller_script} --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        if script:
            controller_cmd += f" --script {script}"
        if args.script_args is not None:
            controller_cmd += f" {' '.join(args.script_args)}"
        control_url = f"localhost:{port}"

    def get_lepton_ip(worker_idx: int) -> str:
        if "LEPTON_JOB_WORKER_INDEX" in os.environ:
            # For non-primary workers, connect to the primary worker (index 0) using its hostname
            prefix = os.environ.get(
                "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
            )
            subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
            hostname = f"{prefix}-{worker_idx}.{subdomain}"
            hostname = resolve_host_blocking(hostname)
        else:
            raise RuntimeError(
                "Lepton job worker index not found in environment variables"
            )
        return hostname

    def get_ip_from_list(worker_idx: int) -> str:
        if args.node_ip_list is not None:
            logger.info(f"Node IP list provided: {args.node_ip_list}")
            ip_list = args.node_ip_list.split(";")
            logger.info(f"Node IP list: {ip_list}")
            if worker_idx < len(ip_list):
                return ip_list[worker_idx]
            else:
                raise RuntimeError(
                    f"Worker index {worker_idx} exceeds the length of the IP list"
                )
        else:
            raise RuntimeError("Node IP list not provided")

    def get_worker_ip(worker_idx: int) -> str:
        if "LEPTON_JOB_WORKER_INDEX" in os.environ:
            return get_lepton_ip(worker_idx)
        elif args.node_ip_list is not None:
            return get_ip_from_list(worker_idx)
        else:
            raise RuntimeError(
                "Replica with GPUs larger than 8 occurs but not on Lepton job, please specify --node-ip-list to provide the IPs of all nodes to enable conenctions to each Rendezvous head node."
            )

    global_launch_settings = replica_placement(
        available_gpus,
        n_policy,
        n_rollouts,
        min_n_gpus_policy,
        min_n_gpus_rollout,
        replica_script,
        control_url,
        output_dir,
        get_worker_ip=get_worker_ip,
        rdzv_port=args.rdzv_port,
        script=script,
        backend=backend,
        config_path=tmpfile_toml,
        script_args=args.script_args,
    )

    num_workers = len(global_launch_settings)
    logger.info(f"Number of workers required: {num_workers}")
    if num_workers > 1:
        logger.info(
            "Multiple worker nodes will be used. Ensure that the launch script is excuted on all worker nodes."
        )
    assert (
        len(available_gpus) * num_workers
        >= min_n_gpus_policy * n_policy + min_n_gpus_rollout * n_rollouts
    ), f"Not enough GPUs available. Required: {min_n_gpus_policy * n_policy + min_n_gpus_rollout * n_rollouts}, Available: {len(available_gpus)}"

    if "LEPTON_JOB_WORKER_INDEX" in os.environ:
        prefix = os.environ.get(
            "LEPTON_JOB_SERVICE_PREFIX", os.environ.get("LEPTON_JOB_NAME")
        )
        subdomain = os.environ.get("LEPTON_SUBDOMAIN", "")
        hostname = f"{prefix}-{cur_work_idx}.{subdomain}"
        import cosmos_rl.utils.network_util as network_util

        ips = network_util.get_eth_ips()
        assert len(ips) > 0, "No IPs found for the current machine"
        logger.info(
            f"Setting hostname to {hostname} {ips[0]} for worker index {cur_work_idx}"
        )
        os.system(f"hostname {ips[0]}")
        if shutil.which("host") is not None:
            # Do a blocking wait until the hostname is properly resolved
            # Only do this if 'host' command is available
            idx = 0
            while idx < num_workers:
                remote_host = f"{prefix}-{idx}.{subdomain}"
                remote_hostname = get_hostname_from_host(remote_host)
                pattern = r"^\d+\.\d+\.\d+\.\d+$"
                if (
                    remote_hostname is not None
                    and re.match(pattern, remote_hostname) is not None
                ):
                    idx = idx + 1
                    continue
                else:
                    logger.info(
                        f"Waiting for hostname {remote_host} to be changed as ready, current: {remote_hostname}"
                    )
                    time.sleep(1)
    if (
        len(global_launch_settings) <= cur_work_idx
        or len(global_launch_settings[cur_work_idx]) == 0
    ):
        if controller_cmd is None:
            logger.info(
                f"No launch settings found for worker index {cur_work_idx}, no need launch"
            )
            sys.exit(0)

    processes = []

    controller_id = -1

    if controller_cmd is not None:
        controller_process = launch_processes(
            [controller_cmd],
            [""],
            [""],
            [
                os.path.join(output_dir, "controller.log")
                if output_dir is not None
                else None
            ],
        )
        controller_id = len(processes)
        processes.append(controller_process[0])

    logger.info(f"Waiting for controller to be ready at {control_url}")
    wait_for_url_ready(
        control_url, controller_process[0] if controller_cmd is not None else None
    )
    logger.info(f"Controller is ready at {control_url}")

    if (
        len(global_launch_settings) > cur_work_idx
        and len(global_launch_settings[cur_work_idx]) != 0
    ):
        commands = global_launch_settings[cur_work_idx][0]
        gpu_devices = global_launch_settings[cur_work_idx][1]
        control_urls = global_launch_settings[cur_work_idx][2]
        output_files = global_launch_settings[cur_work_idx][3]

        # Combine all commands
        logger.info(f"Commands to be executed: {commands}")
        logger.info(f"GPU devices to be used: {gpu_devices}")
        logger.info(f"Control URLs to be used: {control_urls}")
        logger.info(f"Output files: {output_files}")

        # Check if the number of GPU devices matches the number of commands
        assert (
            len(gpu_devices) == len(commands)
        ), f"Number of GPU devices ({len(gpu_devices)}) does not match number of commands ({len(commands)})"

        # Launch all processes
        processes.extend(
            launch_processes(commands, gpu_devices, control_urls, output_files)
        )

    # Wait for all processes to complete without blocking
    while len(processes) > 0:
        for i, process in enumerate(processes):
            try:
                # Check if process has finished without blocking
                if process.poll() is not None:
                    returncode = process.returncode
                    if returncode == 0:
                        logger.info(f"Process {i} completed successfully")
                    else:
                        logger.error(
                            f"Process {i} failed with return code {returncode}"
                        )
                        # Terminate all remaining processes
                        if controller_id == -1 or i == controller_id:
                            for p in processes:
                                try:
                                    p.kill()
                                except Exception as e:
                                    logger.error(f"Error kill process {p}: {e}")
                            logger.error("Terminated all processes due to failure")
                            sys.exit(1)  # Exit with error code 1 if any process failed
                    # Remove completed process from list
                    processes.remove(process)
            except Exception as e:
                logger.error(f"Error monitoring process {i}: {e}")
                # Terminate all remaining processes
                if controller_id == -1 or i == controller_id:
                    for p in processes:
                        try:
                            p.kill()
                        except Exception as e:
                            logger.error(f"Error kill process {p}: {e}")
                    logger.error("Terminated all processes due to error")
                    sys.exit(1)
        # Small sleep to prevent busy waiting
        time.sleep(0.1)

    if tmpfile_toml is not None and os.path.exists(tmpfile_toml):
        # Clean up the temporary file
        try:
            os.unlink(tmpfile_toml)
            tmpfile_toml = None
        except Exception as e:
            logger.error(f"Error deleting temporary file {tmpfile_toml}: {e}")


if __name__ == "__main__":
    main()
