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

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Type, Union, Optional
from transformers import AutoTokenizer
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.tools_use.tool_agent import ToolAgent
from cosmos_rl.dispatcher.data.packer.multi_turn import (
    ConversationType,
    add_assistant_message,
)
from cosmos_rl.utils.logging import logger

import argparse


def worker_entry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the replica entrypoint.")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the web panel on."
    )
    parser.add_argument(
        "--redis-port", type=int, default=12800, help="Port to run the web panel on."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to TOML configuration file to load.",
    )

    parser.add_argument(
        "--redis-logfile-path",
        type=str,
        default="/tmp/redis.log",
        help="The redis server log file path.",
    )
    return parser


class DataPacker(ABC):
    _MODEL_TO_DEFAULT_DATA_PACKER_REGISTRY: Dict[str, Type["DataPacker"]] = {}

    """
    This is where dataset item is transformed into the format required by the rollout engine (e.g. vllm)
    for example:
        - `str` is needed for language model
        - {
            "prompt": prompt,
            "multi_modal_data": {"video": ...},
          } for multi-modal model
    """

    @classmethod
    def register(
        cls,
        model_types: Union[str, List[str]],
        default_data_packer_cls: Type["DataPacker"],
        *,
        allow_override: bool = False,
    ):
        if isinstance(model_types, str):
            model_types = [model_types]
        else:
            model_types = list(model_types)

        for model_type in model_types:
            if (
                not allow_override
                and model_type in DataPacker._MODEL_TO_DEFAULT_DATA_PACKER_REGISTRY
                and DataPacker._MODEL_TO_DEFAULT_DATA_PACKER_REGISTRY[model_type]
                != default_data_packer_cls
            ):
                raise ValueError(f"DataPacker for {model_type} is already registered")
            DataPacker._MODEL_TO_DEFAULT_DATA_PACKER_REGISTRY[model_type] = (
                default_data_packer_cls
            )

    @classmethod
    def get_default_data_packer(cls, model_type: str) -> Type["DataPacker"]:
        if model_type not in DataPacker._MODEL_TO_DEFAULT_DATA_PACKER_REGISTRY:
            raise ValueError(f"DataPacker for {model_type} is not registered")
        return DataPacker._MODEL_TO_DEFAULT_DATA_PACKER_REGISTRY[model_type]()

    def __init__(self, tool_agent: Optional[ToolAgent] = None, *args, **kwargs):
        self.tool_agent = tool_agent

    def setup(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
        *args,
        **kwargs,
    ):
        """
        Called by launcher after being mounted
        """
        assert config is not None, "config should be set"
        assert tokenizer is not None, "tokenizer should be set"
        self.config = config
        self.tokenizer = tokenizer
        if not self.config.rollout.multi_turn_config.enable:
            self.tool_agent = None

        self.custom_chat_template = None
        if self.config.rollout.multi_turn_config.custom_chat_template_path:
            try:
                with open(
                    self.config.rollout.multi_turn_config.custom_chat_template_path, "r"
                ) as f:
                    self.custom_chat_template = f.read()
            except FileNotFoundError:
                logger.warning(
                    f"Custom chat template file not found: {self.config.rollout.multi_turn_config.custom_chat_template_path}, use model default template instead."
                )
                self.custom_chat_template = None

    @abstractmethod
    def get_rollout_input(self, item: Any) -> Any:
        """
        Convert sample to data format required by the rollout engine (e.g. vllm)
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def rollout_collate_fn(self, items: List[Any]) -> List[Any]:
        """
        Collate the rollout inputs into a mini-batch for rollout engine
        """
        return items

    @abstractmethod
    def get_policy_input(
        self,
        sample: Any,
        rollout_output: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """
        Stage for processing samples & rollout output before collating them into a mini-batch
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute the maximum sequence length of the mini-batch
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def policy_collate_fn(
        self,
        processed_samples: List[Any],
        computed_max_len: int,
    ) -> Dict[str, Any]:
        """
        Collate the processed samples into a mini-batch,
        the output will be fed into the policy model for training
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def sft_process_sample(self, sample: Any) -> Any:
        """
        Process the sample into the format required by the SFT model
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass for SFT training"
        )

    def sft_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass for SFT training"
        )

    def sft_collate_fn(
        self,
        sub_batch: List[Dict[str, Any]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        """
        Collate the mini-batch into the format required by the policy model
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass for SFT training"
        )

    def extend_conversation(
        self,
        conversation: ConversationType,
        responses: List[str],
        ground_truth: Optional[str] = None,
    ) -> ConversationType:
        """
        Extend the conversation by models response.
        """
        # By default, we always add response as assistant message
        return add_assistant_message(conversation, "" if responses else responses[0])
