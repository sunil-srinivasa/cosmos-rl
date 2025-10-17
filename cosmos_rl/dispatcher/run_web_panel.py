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
import argparse
import uvicorn
import toml
from fastapi import FastAPI
from contextlib import asynccontextmanager
from torch.utils.data import Dataset
import asyncio
import base64
import cloudpickle
import threading


from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List, Optional, Callable, Union
from cosmos_rl.dispatcher.controller import Controller
import cosmos_rl.utils.constant as constant
from cosmos_rl.dispatcher.protocol import MESH_NAMES
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.protocol import (
    RegisterRequest,
    ErrorResponse,
    RolloutRequest,
    ValidationReportRequest,
    HandshakeInitiatorRequest,
    HandshakeAcceptorRequest,
    UnregisterRequest,
    TrainAckRequest,
    HeartbeatRequest,
    SetProfileRequest,
    SetTracePathRequest,
    NcclErrRequest,
    NcclStoreClearRequest,
    GetShardSendRecvInstsRequest,
)
from cosmos_rl.policy.config import Config as CosmosConfig
import cosmos_rl.utils.util as util
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.constant import COSMOS_ROLLOUT_SCAN_INTERVAL
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_PANEL_SUFFIX,
    COSMOS_API_STATUS_SUFFIX,
    COSMOS_API_META_SUFFIX,
    COSMOS_API_REGISTER_SUFFIX,
    COSMOS_API_SET_PROFILE_SUFFIX,
    COSMOS_API_SET_TRACE_PATH_SUFFIX,
    COSMOS_API_UNREGISTER_SUFFIX,
    COSMOS_API_HEARTBEAT_SUFFIX,
    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX,
    COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX,
    COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX,
    COSMOS_API_NCCL_COMM_ERROR_SUFFIX,
    COSMOS_API_NCCL_COMM_STORE_CLEAR_SUFFIX,
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_ROLLOUT_SUFFIX,
    COSMOS_API_VALIDATION_REPORT_SUFFIX,
    COSMOS_API_POLICY_TRAIN_ACK_SUFFIX,
    COSMOS_API_POLICY_SHARD_INFOS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX,
    COSMOS_API_POLICY_SHARD_SEND_INSTS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX,
    COSMOS_API_GET_TRAINABLE_PARAMS_SUFFIX,
)
from cosmos_rl.dispatcher.data.packer.base import DataPacker, worker_entry_parser
from cosmos_rl.utils.payload import extract_rollouts
from fastapi.responses import Response
from fastapi import Request
from concurrent.futures import ThreadPoolExecutor


def create_error_response(
    code: int, message: str, status_code: Optional[int] = None
) -> JSONResponse:
    if status_code is None:
        status_code = code // 100
    return JSONResponse(
        ErrorResponse(message=message, code=code).model_dump(), status_code=status_code
    )


controller = Controller()
server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    shutdown_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()

    def monitor_replica_status():
        while not shutdown_event.is_set():
            # Run in separate process
            controller.policy_status_manager.maintain_life_status()
            controller.rollout_status_manager.maintain_life_status(
                controller.policy_status_manager
            )
            if shutdown_event.wait(timeout=COSMOS_ROLLOUT_SCAN_INTERVAL):
                break  # Exit early if shutdown signaled during sleep

    task = loop.run_in_executor(executor, monitor_replica_status)
    yield
    # Signal shutdown
    shutdown_event.set()
    await task


app = FastAPI(lifespan=lifespan)


@app.get(COSMOS_API_PANEL_SUFFIX)
async def panel():
    # HTML template with JavaScript for auto-refresh
    with open(
        os.path.join(
            os.path.dirname(__file__), "config/frontend", "dispatcher_status.html"
        ),
        "r",
        encoding="utf-8",
    ) as file:
        html = file.read()
    return HTMLResponse(html)


"""
API for replica-controller communication
"""


@app.get(COSMOS_API_STATUS_SUFFIX)
async def get_status():
    return {
        "mesh_names": MESH_NAMES,
        "policy_replicas": _serialize_replicas(
            controller.policy_status_manager.policy_replicas
        ),
        "rollout_replicas": _serialize_replicas(
            controller.rollout_status_manager.rollout_replicas
        ),
    }


@app.get(COSMOS_API_META_SUFFIX)
async def meta():
    meta = {
        "config": controller.config,
    }
    if controller.user_data_packer is not None:
        meta["user_data_packer"] = base64.b64encode(
            cloudpickle.dumps(controller.user_data_packer)
        ).decode("utf-8")
    if controller.user_val_data_packer is not None:
        meta["user_val_data_packer"] = base64.b64encode(
            cloudpickle.dumps(controller.user_val_data_packer)
        ).decode("utf-8")
    return meta


@app.post(COSMOS_API_REGISTER_SUFFIX)
async def register(request: RegisterRequest):
    try:
        await controller.register(
            Atom.from_register_request(request),
            role=request.role,
        )
        return {"message": "Registered"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post(COSMOS_API_UNREGISTER_SUFFIX)
async def unregister(request: UnregisterRequest):
    try:
        await controller.unregister(request.replica_name)
    except Exception as e:
        logger.error(f"[Controller] Unregister failed: {e}")
    finally:
        if (
            (
                controller.policy_status_manager.training_finished()
                or not controller.is_rl
            )
            and len(controller.policy_status_manager) == 0
            and len(controller.rollout_status_manager) == 0
        ):
            logger.info("[Controller] All replicas are finished, finalizing...")
            global server
            server.should_exit = True
        return {"message": "Unregistered"}


@app.post(COSMOS_API_SET_PROFILE_SUFFIX)
async def set_profile(request: SetProfileRequest):
    logger.info(f"[Dispatcher] set profile request: {request}")
    msg = await controller.set_profile(request)
    return msg


@app.post(COSMOS_API_SET_TRACE_PATH_SUFFIX)
async def set_trace_path(request: SetTracePathRequest):
    atom = await controller.set_trace_path(
        request.replica_name, request.trace_path, request.global_rank
    )
    if atom is not None:
        return {"message": f"Trace path set for atom: {atom}"}
    else:
        return {"message": "Ignore the trace path request!"}


@app.post(COSMOS_API_HEARTBEAT_SUFFIX)
async def heartbeat(request: HeartbeatRequest):
    # Set the replica timestamp to the current time for heartbeat
    controller.replica_heartbeat(request.replica_name)
    return {"message": "Heartbeat received"}


@app.post(COSMOS_API_POLICY_SHARD_INFOS_SUFFIX)
async def policy_shard_infos(request: Request):
    content_type = request.headers.get("Content-Type")
    if content_type != "application/msgpack":
        return create_error_response(
            constant.ErrorCode.INVALID_REQUEST,
            "Invalid Content-Type, expected application/msgpack",
        )

    raw_bytes = await request.body()
    await controller.policy_to_rollout_shard_mapper.set_shard_infos_of_policy(
        raw_bytes,
        controller.policy_status_manager.n_atoms_per_replica(),
    )
    return {"message": "Policy shard infos set"}


@app.post(COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX)
async def rollout_shard_infos(request: Request):
    content_type = request.headers.get("Content-Type")
    if content_type != "application/msgpack":
        return create_error_response(
            constant.ErrorCode.INVALID_REQUEST,
            "Invalid Content-Type, expected application/msgpack",
        )

    raw_bytes = await request.body()
    await controller.policy_to_rollout_shard_mapper.set_shard_infos_of_rollout(
        raw_bytes,
        controller.rollout_status_manager.n_atoms_per_replica(),
    )
    return {"message": "Rollout shard infos set"}


@app.post(COSMOS_API_POLICY_SHARD_SEND_INSTS_SUFFIX)
async def policy_shard_send_insts(request: GetShardSendRecvInstsRequest):
    """
    Get the send instructions for policy.
    :return: A list of send instructions for policy.
    """
    logger.debug(
        f"[Dispatcher] Get policy shard send instructions for rank {request.rank}"
    )
    await controller.policy_to_rollout_shard_mapper.scheme_generation_done.wait()
    # Get the send instructions for policy
    send_insts = (
        await controller.policy_to_rollout_shard_mapper.get_send_insts_for_policy(
            request.rank
        )
    )
    # If the send instructions are not found, return an error response
    if send_insts is None:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR,
            "Policy shard send instructions not found",
        )
    logger.debug(
        f"[Dispatcher] Received policy shard send instructions for rank {request.rank}"
    )
    return Response(content=send_insts, media_type="application/msgpack")


@app.post(COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX)
async def rollout_shard_recv_insts(request: GetShardSendRecvInstsRequest):
    """
    Get the receive instructions for rollout.
    :return: A list of receive instructions for rollout.
    """
    logger.debug(
        f"[Dispatcher] Get rollout shard receive instructions for rank {request.rank}"
    )
    # Wait for the scheme generation to be done
    await controller.policy_to_rollout_shard_mapper.scheme_generation_done.wait()
    # Get the receive instructions for rollout
    recv_insts = (
        await controller.policy_to_rollout_shard_mapper.get_recv_insts_for_rollout(
            request.rank
        )
    )
    # If the receive instructions are not found, return an error response
    if recv_insts is None:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR,
            "Rollout shard receive instructions not found",
        )

    logger.debug(
        f"[Dispatcher] Received rollout shard receive instructions for rank {request.rank}"
    )

    return Response(content=recv_insts, media_type="application/msgpack")


@app.get(COSMOS_API_GET_TRAINABLE_PARAMS_SUFFIX)
async def get_trainable_params():
    try:
        return {
            "trainable_params": list(
                controller.policy_to_rollout_shard_mapper.trainable_params
            )
        }
    except Exception:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR,
            "Error getting trainable params",
        )


"""
NCCL Handshake API
"""


@app.post(COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX)
async def comm_initiator(request: HandshakeInitiatorRequest):
    if request.handle_base64 is None or request.handle_base64 == "":
        return create_error_response(
            constant.ErrorCode.INVALID_REQUEST, "Handle is required"
        )

    await controller.update_kv_store(request.unique_pair_name, request.handle_base64)
    return {"message": "Handshake initiator received"}


@app.post(COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX)
async def comm_acceptor(request: HandshakeAcceptorRequest):
    if request.unique_pair_name not in controller.temp_kv_store:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR, "Unique pair name not found"
        )
    return {"handle_base64": controller.temp_kv_store.get(request.unique_pair_name)}


@app.post(COSMOS_API_NCCL_COMM_ERROR_SUFFIX)
async def comm_error(request: NcclErrRequest):
    await controller.set_replica_ncclerror(request.replica_name, request.error)
    return {"message": "DetectTimeout received"}


@app.post(COSMOS_API_NCCL_COMM_STORE_CLEAR_SUFFIX)
async def comm_store_clear(request: NcclStoreClearRequest):
    try:
        await controller.clear_temp_kv_store(request.unique_pair_name)
    except Exception as e:
        logger.error(f"[Controller] Error clearing store: {e}")
    return {"message": "Store cleared"}


@app.get(COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX)
async def comm_get_all():
    return {"comm_info": controller.temp_kv_store}


"""
Rollout API
"""


@app.get(COSMOS_API_NEXT_PROMPT_SUFFIX)
async def get_batched_prompt(n: int, validation_step: Optional[int] = None):
    prompt_id_and_payload_list, is_end = await controller.get_batched_prompt(
        n, validation_step
    )
    return {
        "prompt_id_and_payload_list": prompt_id_and_payload_list,
        "is_end": is_end,
    }


@app.post(COSMOS_API_VALIDATION_REPORT_SUFFIX)
async def validation_report(request: ValidationReportRequest):
    rollouts_list, invalid_rollouts_list = extract_rollouts(
        request.payloads, True, request.prompt_idxs
    )
    assert len(invalid_rollouts_list) == 0, "Validation rollouts should all be valid"
    controller.policy_status_manager.validation_report_validation_results(
        request.validation_step, rollouts_list, controller.rollout_status_manager
    )
    return {"message": "Validation rollout put"}


@app.post(COSMOS_API_ROLLOUT_SUFFIX)
async def put_rollout_group(rollout: RolloutRequest):
    try:
        if rollout.is_end:
            assert (
                len(rollout.prompt_idxs) == 0
            ), "Prompt idxs should be empty if is_end is True"
            logger.info(
                f"[Controller] Received rollout end signal from {rollout.src_replica_name}"
            )
            controller.rollout_status_manager.rollout_end(rollout.src_replica_name)
            if controller.rollout_status_manager.all_rollouts_ended():
                total_pending_rollouts = (
                    controller.policy_status_manager.total_pending_rollouts()
                )
                logger.info(
                    f"[Controller] All rollouts have ended, recompute total steps with {total_pending_rollouts} remaining rollouts..."
                )
                original_total_steps = controller.policy_status_manager.total_steps
                controller.policy_status_manager.recompute_total_steps(
                    explicit_num_remaining_samples=total_pending_rollouts
                )
                new_total_steps = controller.policy_status_manager.total_steps
                if new_total_steps > controller.policy_status_manager.current_step:
                    logger.info(
                        "[Controller] There are still remaining steps, no op required"
                    )
                    # There are still remaining steps, no op required
                    pass
                else:
                    if (
                        controller.policy_status_manager.current_step
                        == original_total_steps
                    ):
                        logger.info(
                            "[Controller] No remaining steps, policy and rollouts happen to finish at the same time"
                        )
                        # No remaining steps, policy and rollouts happen to finish at the same time
                        pass
                    else:
                        logger.info(
                            "[Controller] Clear the rollout buffer, and trigger an extra `DataFetch`"
                        )
                        # Clear the rollout buffer
                        controller.policy_status_manager.rollout_buffer.queue.clear()
                        controller.policy_status_manager.total_steps = (
                            controller.policy_status_manager.current_step + 1
                        )

                        # Trigger an extra `DataFetch & P2R/R2R`
                        controller.policy_status_manager.try_trigger_data_fetch_and_training(
                            is_fake_last_cmd=True
                        )

            return {"message": "Rollout end signal received"}

        # Dynamic Sampling: Filter out the rollouts that the rewards are all the same
        valid_rollouts_list, invalid_rollouts_list = extract_rollouts(
            rollout.payloads, rollout.is_end, rollout.prompt_idxs
        )
        # Flatten the rollouts into a single list
        valid_rollouts = [
            rollout
            for rollouts_group in valid_rollouts_list
            for rollout in rollouts_group  # rollouts_group: all rollouts of the same prompt.
        ]
        invalid_rollouts = [
            rollout
            for rollouts_group in invalid_rollouts_list
            for rollout in rollouts_group
        ]

        if len(valid_rollouts) > 0:
            logger.debug(
                f"[RolloutGroup] from replica: {rollout.src_replica_name} with {len(rollout.payloads)} samples:"
                f"example: rollouts[0]\n{valid_rollouts[0]}"
            )

        await controller.put_rollouts(valid_rollouts, invalid_rollouts)
        return {"message": "Rollout put"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post(COSMOS_API_POLICY_TRAIN_ACK_SUFFIX)
async def train_ack(request: TrainAckRequest):
    try:
        replicaname = request.replica_name
        step = request.weight_step
        total_steps = request.total_steps
        profile_finished = request.profile_finished
        report_data = request.report_data
        controller.policy_status_manager.train_ack(
            replicaname,
            step,
            total_steps,
            profile_finished,
            report_data,
            controller.rollout_status_manager,
        )
        return {"message": "Ack completed"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


def _serialize_replicas(replicas: Dict[str, Replica]) -> List[Dict]:
    result = []
    for name, replica in replicas.items():
        result.append(replica.to_dict())
    return result


def main(
    dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
    data_packer: Optional[DataPacker] = None,
    reward_fns: Optional[List[Callable]] = None,
    filter_reward_fns: Optional[List[Callable]] = None,
    val_dataset: Optional[Dataset] = None,
    val_reward_fns: Optional[List[Callable]] = None,
    val_data_packer: Optional[DataPacker] = None,
    custom_logger_fns: Optional[List[Callable]] = None,
    sampler: Optional[Callable] = None,
    batch_sampler: Optional[Callable] = None,
    val_sampler: Optional[Callable] = None,
    val_batch_sampler: Optional[Callable] = None,
    args: Optional[argparse.Namespace] = None,
    **kwargs,
):
    if kwargs:
        logger.warning(
            f"Params: {list(kwargs.keys())} are not being used in controller initialization."
        )

    # Deprecated: The following code is to ensure backward compatibility:
    # where `dispatcher` is always launched in custom script
    role = os.environ.get("COSMOS_ROLE")
    assert role in ["Policy", "Rollout", "Controller"], f"Invalid role: {role}"
    if role == "Controller":
        pass
    else:
        logger.warning(
            "Deprecated: Please update your script to use `cosmos_rl.launcher.launch()` instead of `cosmos_rl.dispatcher.run_web_panel.main`"
        )
        if role == "Policy":
            from cosmos_rl.policy.train import main as policy_main

            policy_main(
                dataset=dataset,
                data_packer=data_packer,
                val_dataset=val_dataset,
                val_data_packer=val_data_packer,
                sampler=sampler,
                batch_sampler=batch_sampler,
                val_sampler=val_sampler,
                val_batch_sampler=val_batch_sampler,
            )
        else:
            from cosmos_rl.rollout.rollout_entrance import run_rollout

            run_rollout(
                dataset=dataset,
                reward_fns=reward_fns,
                filter_reward_fns=filter_reward_fns,
                val_dataset=val_dataset,
                val_reward_fns=val_reward_fns,
            )
        return

    if args is None:
        # This means that args are not parsed in dataset entry script
        # So we need to parse the args manually
        parser = worker_entry_parser()
        try:
            args = parser.parse_args()
        except SystemExit as e:
            logger.error(
                "Error when parsing args. Did you use custom arguments in your script? If so, please check your custom script and pass `args` to this main function."
            )
            raise e

    # Load config from file if provided
    loaded_config = None
    assert os.path.exists(args.config), f"Config file {args.config} does not exist."

    try:
        logger.info(f"Attempting to load configuration from {args.config}")
        with open(args.config, "r") as f:
            config_dict = toml.load(f)

        # Ensure CosmosConfig is available (it's imported at the top now)
        # from cosmos_rl.policy.config import Config as CosmosConfig
        # Need SFTDataConfig and GrpoConfig for from_dict

        loaded_config = CosmosConfig.from_dict(config_dict)
        # Use redis port from config if available, otherwise use arg/default
        if hasattr(loaded_config, "redis") and loaded_config.redis:
            try:
                redis_port_from_config = int(loaded_config.redis)
                args.redis_port = redis_port_from_config
                logger.info(f"Using Redis port {args.redis_port} from config file.")
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid redis port format in config file: {loaded_config.redis}. Using default/arg: {args.redis_port}"
                )

        if data_packer is not None:
            assert isinstance(
                data_packer, DataPacker
            ), "data_packer should be a DataPacker instance"
        controller.setup(
            loaded_config,
            redis_port=args.redis_port,
            redis_logfile_path=args.redis_logfile_path,
            dataset=dataset,
            data_packer=data_packer,
            val_dataset=val_dataset,
            val_data_packer=val_data_packer,
            custom_logger_fns=custom_logger_fns,
            sampler=sampler,
            batch_sampler=batch_sampler,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
        )
        logger.info(f"Successfully loaded configuration from {args.config}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {args.config}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or parse config file {args.config}: {e}.",
            exc_info=True,
        )

    config = uvicorn.Config(
        app, host="0.0.0.0", port=util.find_available_port(args.port), access_log=False
    )
    global server
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
