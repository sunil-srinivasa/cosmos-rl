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

import re
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify.errors import TimeoutException
from transformers import PreTrainedTokenizer
from cosmos_rl.policy.config import Config
from typing import Union, Callable, Tuple
from cosmos_rl.utils.constant import RewardFn
from cosmos_rl.utils.logging import logger
from typing import Dict, Optional, List
from cosmos_rl.dispatcher.data.packer import DataPacker

math_comparer = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def format_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
    try:
        pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(pattern, to_be_evaluated, re.DOTALL)
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            return 0.0
        else:
            return 1.0
    except Exception as e:  # noqa: BLE001
        logger.debug("Exception in format_reward_func: %s", e)
        return 0.0


def overlong_reward_fn(
    to_be_evaluated: str,
    reference: Union[str, None],
    config: Config,
    tokenizer: PreTrainedTokenizer,
    **kwargs,
) -> float:
    """
    Reward function that checks if the completion is too long (DAPO).
    If the completion is longer than threshold, adopt a soft punishment.
    """
    overlong_buffer_len = config.train.train_policy.overlong_reward.buffer_length
    expected_len = config.rollout.max_response_length - overlong_buffer_len
    valid_response_length = len(
        tokenizer.encode(to_be_evaluated, add_special_tokens=False)
    )
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = config.train.train_policy.overlong_reward.penalty_factor
    overlong_reward = min(
        -exceed_len / overlong_buffer_len * overlong_penalty_factor, 0
    )
    return overlong_reward


def direct_math_reward_fn(
    to_be_evaluated: str,
    reference: Union[str, None],
    **kwargs,
) -> float:
    """
    Compute the reward for the `to_be_evaluated` and `reference`.
    The reward is 1 if the `to_be_evaluated` is correct, otherwise -1.
    Answer pattern can be customized.
    """
    answer_pattern = kwargs.get("answer_pattern", r"(?i)Answer\s*:\s*([^\n]+)")
    # Extract answer from solution
    match = re.findall(answer_pattern, to_be_evaluated)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    if pred == reference:
        return 1.0
    else:
        return -1.0


def boxed_math_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
    """
    Compute the reward for the `to_be_evaluated` and `reference`.
    The reward is 1 if the `to_be_evaluated` is correct, otherwise 0.
    Answer are supposed to be in format: `\boxed{...}`.
    """
    try:
        score, _ = math_comparer([reference], [to_be_evaluated])
        return score
    except TimeoutException as e:
        logger.error(
            f"Caught TimeoutException: {e}\nreference={reference}\nto_be_evaluated={to_be_evaluated}"
        )
        return 0.0
    except Exception:
        return 0.0


def single_choice_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    reward = 0.0
    try:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r"<answer>(.*?)</answer>", reference, re.DOTALL)
        ground_truth = sol_match.group(1).strip() if sol_match else reference.strip()

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r"<answer>(.*?)</answer>", to_be_evaluated, re.DOTALL)
        student_answer = (
            content_match.group(1).strip() if content_match else to_be_evaluated.strip()
        )

        # Compare the extracted answers
        if student_answer.lower() == ground_truth.lower():
            reward = 1.0
    except Exception:
        reward = 0.0

    return reward


def gsm8k_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
    def extract_solution(solution_str, method="strict"):
        assert method in ["strict", "flexible"]

        if method == "strict":
            # this also tests the formatting of the model
            solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
            if solution is None:
                final_answer = None
            else:
                final_answer = solution.group(0)
                final_answer = (
                    final_answer.split("#### ")[1].replace(",", "").replace("$", "")
                )
        elif method == "flexible":
            answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
            final_answer = None
            if len(answer) == 0:
                # no reward is there is no answer
                pass
            else:
                invalid_str = ["", "."]
                # find the last number that is not '.'
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
        return final_answer

    try:
        return extract_solution(to_be_evaluated) == extract_solution(reference)
    except Exception:
        return 0.0


REWARD_FUNC_MAPPING = {
    RewardFn.DIRECT_MATH: direct_math_reward_fn,
    RewardFn.BOXED_MATH: boxed_math_reward_fn,
    RewardFn.SINGLE_CHOICE: single_choice_reward_fn,
    RewardFn.GSM8K: gsm8k_reward_fn,
    RewardFn.FORMAT: format_reward_fn,
    RewardFn.OVERLONG: overlong_reward_fn,
}


class Reward:
    def __init__(
        self,
        config: Config,
        tokenier: PreTrainedTokenizer,
        reward_function: Optional[Dict[str, float]] = None,
        explicit_reward_fn: Optional[List[Callable]] = None,
        explicit_filter_reward_fn: Optional[List[Callable]] = None,
        data_packer: Optional[DataPacker] = None,
    ):
        self.config = config
        self.tokenizer = tokenier
        if explicit_filter_reward_fn is None:
            explicit_filter_reward_fn = config.train.train_policy.filter_reward_metric
        if reward_function is None:
            reward_function = config.train.train_policy.reward_function

        self.filter_reward_fns = []
        index_for_filter_in_explicit = []
        filter_reward_name_to_weight = {}
        if explicit_filter_reward_fn is not None:
            explicit_filter_reward_fn = (
                explicit_filter_reward_fn
                if isinstance(explicit_filter_reward_fn, list)
                else [explicit_filter_reward_fn]
            )
            for item in explicit_filter_reward_fn:
                if isinstance(item, tuple):
                    fn, weight = item
                else:
                    fn = item
                    weight = 1.0
                if isinstance(fn, str):
                    if fn not in REWARD_FUNC_MAPPING:
                        raise ValueError(f"Filtered reward function {fn} not found.")
                    if fn not in reward_function:
                        self.filter_reward_fns.append((REWARD_FUNC_MAPPING[fn], weight))
                    else:
                        filter_reward_name_to_weight[fn] = weight
                elif isinstance(fn, int):
                    assert (
                        explicit_reward_fn is not None
                    ), "When filtered reward function is given as index, explicit_reward_fn must be provided."
                    index_for_filter_in_explicit.append((fn, weight))
                elif isinstance(fn, Callable):
                    self.filter_reward_fns.append((fn, weight))
                else:
                    raise ValueError(
                        f"Filtered reward function must be str, int or Callable, but got {type(fn)}"
                    )

        if explicit_reward_fn:
            self.reward_funcs = (
                explicit_reward_fn
                if isinstance(explicit_reward_fn, list)
                else [explicit_reward_fn]
            )
            logger.info(
                f"[Reward] Using provided reward functions: {self.reward_funcs}, `config.train.train_policy.reward_function` will be ignored"
            )
            self.is_filter = [0.0] * len(self.reward_funcs)
            for i, weight in index_for_filter_in_explicit:
                assert (
                    i < len(self.reward_funcs)
                ), f"Index {i} for filtered reward function is out of range, only {len(self.reward_funcs)} reward functions are provided."
                self.is_filter[i] = weight
        else:
            self.reward_funcs = []
            self.is_filter = []
            for name, weight in reward_function.items():
                reward_func = RewardFn.from_string(name)
                if reward_func not in REWARD_FUNC_MAPPING:
                    raise ValueError(
                        f"Reward function {reward_func} not found in mapping."
                    )
                self.is_filter.append(filter_reward_name_to_weight.get(name, 0.0))
                self.reward_funcs.append((REWARD_FUNC_MAPPING[name], weight))
                logger.info(f"[Reward] Using reward functions: {reward_function}")
        logger.info(
            f"[Reward] Using filtered reward functions: {self.filter_reward_fns}"
        )
        logger.info(f"[Reward] is_filter: {self.is_filter}")
        self.data_packer = data_packer

    def compute_reward(
        self,
        to_be_evaluated: str,
        reference: Union[str, None],
        prompt: Union[str, List] = "",
        **kwargs,
    ) -> Tuple[float, float]:
        total_reward = 0.0
        filter_reward = 0.0
        for x, filter in zip(self.reward_funcs, self.is_filter):
            if isinstance(x, tuple):
                func, weight = x
            else:
                func = x
                weight = 1.0
            val = func(
                to_be_evaluated,
                reference,
                prompt=prompt,
                data_packer=self.data_packer,
                config=self.config,
                tokenizer=self.tokenizer,
            )
            total_reward += weight * val
            filter_reward += filter * val

        for func, weight in self.filter_reward_fns:
            val = func(
                to_be_evaluated,
                reference,
                prompt=prompt,
                data_packer=self.data_packer,
                config=self.config,
                tokenizer=self.tokenizer,
            )
            filter_reward += val * weight

        if all([f == 0.0 for f in self.is_filter]) and len(self.filter_reward_fns) == 0:
            filter_reward = total_reward
        return total_reward, filter_reward
