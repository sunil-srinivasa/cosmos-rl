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
from typing import Any, List, Tuple


REGISTERED_ALGOs = {}


class RuleBasedAlgo(ABC):
    @abstractmethod
    def compute_reward(
        self, to_be_evaluated: Any, reference: Any, prompt: Any = None
    ) -> Tuple[float, float]:
        return 0.0, 0.0

    @abstractmethod
    def compute_advantage(self, rewards: List[float]) -> List[float]:
        return rewards

    @abstractmethod
    def ready(self) -> bool:
        return False


def _register_rule_based_algo(name: str, algo: RuleBasedAlgo):
    REGISTERED_ALGOs[name] = algo
