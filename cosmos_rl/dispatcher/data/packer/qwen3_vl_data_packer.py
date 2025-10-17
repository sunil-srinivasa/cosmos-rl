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

from cosmos_rl.dispatcher.data.packer.base import DataPacker
from typing import List, Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from cosmos_rl.utils.util import retry
from cosmos_rl.policy.config import Config
from cosmos_rl.dispatcher.data.schema import ChatMessage
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from PIL import Image
import base64
import io

IGNORE_LABEL_ID = -100


def process_vision_info(sample: List[Dict[str, Any]]) -> Tuple[Any, Any]:
    image_inputs = []
    video_inputs = []
    for x in sample:
        if x["role"] == "user":
            for item in x["content"]:
                if item["type"] == "image":
                    image_inputs.append(item["image"])
                if item["type"] == "video":
                    video_inputs.append(item["video"])
    return image_inputs, video_inputs


def encode_image_to_base64(image_inputs: List[str]) -> List[str]:
    new_image_inputs = []
    for image_input in image_inputs:
        img_bytes = base64.b64decode(image_input)
        img_buffer = io.BytesIO(img_bytes)
        image = Image.open(img_buffer)
        new_image_inputs.append(image)
    return new_image_inputs


class Qwen3_VL_DataPacker(DataPacker):
    """
    Data protocol & processing logic for the Qwen3.5 VLMs for SFT and RL training.
    """

    Payload = List[Dict[str, Any]]

    class RLPolicyInput:
        input_ids: List[int]
        logprob_masks: List[int]

        def __init__(self, input_ids: List[int], logprob_masks: List[int]):
            self.input_ids = input_ids
            self.logprob_masks = logprob_masks

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        super().setup(config, tokenizer, *args, **kwargs)
        self.hf_processor = retry(AutoProcessor.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )

        hf_config = retry(AutoConfig.from_pretrained)(
            config.policy.model_name_or_path, trust_remote_code=True
        )

        image_token_id = getattr(hf_config, "image_token_id", None) or getattr(
            hf_config.vision_config, "image_token_id", None
        )
        assert image_token_id is not None, f"Cannot find image token id in {hf_config=}"
        self.image_token_id = image_token_id
        self.image_token = getattr(self.hf_processor, "image_token", None)

        if self.image_token is None:
            self.image_token = self.tokenizer.decode([image_token_id])

        video_token_id = getattr(hf_config, "video_token_id", None) or getattr(
            hf_config.vision_config, "video_token_id", None
        )
        if video_token_id is None:
            self.video_token = None
            self.video_token_id = None
        else:
            self.video_token = self.tokenizer.decode([video_token_id])
            self.video_token_id = video_token_id
        self.vision_ids = [self.image_token_id, self.video_token_id]
        self.hf_config = hf_config

    def get_rollout_input(self, sample: Payload) -> Any:
        """
        This VL data packer only accepts the conversation data format.
        check https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat for more details.

        example:
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        It is user's responsibility to ensure the conversation format is correct
          and multi-media files involved in conversation are accessible.
        """
        sample = [x.model_dump() if isinstance(x, ChatMessage) else x for x in sample]
        assert all(
            isinstance(x, dict) and "role" in x and "content" in x for x in sample
        ), "All samples should be in conversation format, but got: {}".format(sample)

        if self.image_token is not None:
            for x in sample:
                if x["role"] == "user":
                    contents = x["content"]
                    for idx, content in enumerate(contents):
                        if (
                            content["type"] == "text"
                            and self.image_token in content["text"]
                        ):
                            new_content = content.copy()
                            contents[idx]["text"] = new_content["text"].replace(
                                self.image_token, ""
                            )

        # Here we need to convert the conversation format to the format required by vllm
        prompt = self.hf_processor.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(sample)
        # TODO: add video support
        if len(video_inputs) > 0:
            return {
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs},
                "mm_processor_kwargs": {},
            }
        elif len(image_inputs) > 0:
            assert len(image_inputs) == 1, f"{len(image_inputs)=}"
            return {
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            }
        else:
            return {
                "prompt": prompt,
            }

    def _replace_assistant_content(
        self,
        token_ids: List[int],
        label_ids: List[int],
        pad_token_id: int,
        eos_token_id: int,
        replacement_ids: List[int],
        pad_run_length: int = 10,
    ) -> List[int]:
        """
        Find the first run of exactly `pad_run_length` pad_token_id's in token_ids,
        replace that run with replacement_ids, and return the new list.
        If no such run is found, returns the original list unchanged.
        """
        n = len(token_ids)
        target_run = [pad_token_id] * pad_run_length

        # find the start index of the first matching run
        for i in range(n - pad_run_length + 1):
            if token_ids[i : i + pad_run_length] == target_run:
                # splice in the replacement
                if (
                    len(token_ids) > i + pad_run_length
                    and token_ids[i + pad_run_length] == eos_token_id
                ):
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + [eos_token_id]
                        + label_ids[i + pad_run_length + 1 :]
                    )
                else:
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + label_ids[i + pad_run_length :]
                    )
                return (
                    True,
                    token_ids[:i] + replacement_ids + token_ids[i + pad_run_length :],
                    label_ids,
                )
        # no match found
        return False, token_ids, label_ids

    def _get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VLMoe use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        image_token_id = self.hf_config.image_token_id
        video_token_id = self.hf_config.video_token_id
        vision_start_token_id = self.hf_config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _process_single_sample(
        self,
        conversation: "Qwen3_VL_DataPacker.Payload",
        add_generation_prompt: bool,
    ) -> Dict[str, Any]:
        try:
            # Replace all the assistant content with consecutive `pad_token` * 10
            pad_token = self.tokenizer.pad_token
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_run_length = 10
            assistant_contents = []
            messages = None
            # SFT
            if "messages" in conversation:
                messages = conversation["messages"]
                for message in messages:
                    if message["role"] == "assistant":
                        content = message["content"]
                        new_content = content.copy()
                        if isinstance(new_content, str):
                            assistant_contents.append(new_content)
                            new_content = pad_token * pad_run_length
                        elif isinstance(new_content, dict):
                            assert (
                                "text" in new_content
                            ), f"text not in content: {content}"
                            assistant_contents.append(new_content["text"])
                            new_content["text"] = pad_token * pad_run_length
                        elif isinstance(content, list):
                            for i, item in enumerate(content):
                                if isinstance(item, dict):
                                    assert (
                                        "text" in item
                                    ), f"text not in content: {item}"
                                    assistant_contents.append(item["text"])
                                    new_content[i]["text"] = pad_token * pad_run_length
                                else:
                                    raise ValueError(
                                        f"Unsupported content type: {type(item)}"
                                    )
                        else:
                            raise ValueError(
                                f"Unsupported content type: {type(content)}"
                            )
                        message["content"] = new_content
            else:
                # RL
                messages = conversation
                if self.image_token is not None:
                    for x in messages:
                        if x["role"] == "user":
                            contents = x["content"]
                            for idx, content in enumerate(contents):
                                if (
                                    content["type"] == "text"
                                    and self.image_token in content["text"]
                                ):
                                    new_content = content.copy()
                                    contents[idx]["text"] = new_content["text"].replace(
                                        self.image_token, ""
                                    )

            text = self.hf_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if "images" in conversation:
                image_inputs = conversation["images"]
            else:
                image_inputs, video_inputs = process_vision_info(conversation)
                assert all(
                    (isinstance(x, str) for x in image_inputs)
                ), f"{image_inputs=}"
                assert (
                    len(video_inputs) == 0
                ), "Currently video input is not supported for HF VLM"
                image_inputs = encode_image_to_base64(image_inputs)

            kwarg = {
                "return_tensors": "pt",
                "images": image_inputs,
            }
            inputs = self.hf_processor(
                text=[text],
                **kwarg,
            )
            input_ids = inputs["input_ids"][0].tolist()
            label_ids = [IGNORE_LABEL_ID] * len(input_ids)

            for assistant_content in assistant_contents:
                replacement_ids = self.tokenizer.encode(
                    assistant_content, add_special_tokens=False
                )

                replaced, input_ids, label_ids = self._replace_assistant_content(
                    input_ids,
                    label_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    replacement_ids=replacement_ids,
                    pad_run_length=pad_run_length,
                )
                if not replaced:
                    raise ValueError("No assistant content to replace")
                if len(input_ids) != len(label_ids):
                    raise ValueError(
                        f"input_ids and label_ids should have the same length, but got {len(input_ids)} and {len(label_ids)}"
                    )
        except Exception as e:
            print(f"Error processing sample: {e}, please fix to ensure SFT works")
            raise e

        result_dict = {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }
        if "pixel_values_videos" in inputs:
            result_dict["pixel_values_videos"] = inputs["pixel_values_videos"]
            if "video_grid_thw" in inputs:
                result_dict["video_grid_thw"] = inputs["video_grid_thw"]
            else:
                result_dict["video_grid_thw"] = None

        if "pixel_values" in inputs:
            result_dict["pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                result_dict["image_grid_thw"] = inputs["image_grid_thw"]
            else:
                result_dict["image_grid_thw"] = None

        # position_ids: (3, 1, seq_len)
        # Only for Qwen3VLMOE
        position_ids, _ = self._get_rope_index(
            input_ids=torch.tensor(input_ids).unsqueeze(0).clone(),
            image_grid_thw=torch.tensor(result_dict.get("image_grid_thw"))
            if "image_grid_thw" in result_dict
            else None,
            video_grid_thw=torch.tensor(result_dict.get("video_grid_thw"))
            if "video_grid_thw" in result_dict
            else None,
            attention_mask=None,
        )
        result_dict["position_ids"] = position_ids.clone()

        return result_dict

    def _collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        pixel_values_videos = [x["pixel_values_videos"] for x in processed_samples]
        video_grid_thw = [x["video_grid_thw"] for x in processed_samples]
        pixel_values = [x["pixel_values"] for x in processed_samples]
        image_grid_thw = [x["image_grid_thw"] for x in processed_samples]

        if all([x is not None for x in pixel_values_videos]):
            assert all(
                [x is not None for x in pixel_values_videos]
            ), "pixel_values_videos should not be None"
            max_len = max([x.shape[0] for x in pixel_values_videos])
            for i in range(len(pixel_values_videos)):
                pixel_values_videos[i] = pixel_values_videos[i].unsqueeze(0)
                assert (
                    pixel_values_videos[i].ndim == 3
                ), f"pixel_values_videos[i].ndim: {pixel_values_videos[i].ndim}"
                pixel_values_videos[i] = F.pad(
                    pixel_values_videos[i],
                    (0, 0, 0, max_len - pixel_values_videos[i].shape[1]),
                )
            pixel_values_videos = torch.cat(pixel_values_videos, dim=0)
        else:
            assert all(
                [x is None for x in pixel_values_videos]
            ), "pixel_values_videos should be None"
            pixel_values_videos = None

        if all([x is not None for x in video_grid_thw]):
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            assert all(
                [x is None for x in video_grid_thw]
            ), "video_grid_thw should be None"
            video_grid_thw = None

        if all([x is not None for x in pixel_values]):
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            assert all([x is None for x in pixel_values]), "pixel_values should be None"
            pixel_values = None

        if all([x is not None for x in image_grid_thw]):
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
        else:
            assert all(
                [x is None for x in image_grid_thw]
            ), "image_grid_thw should be None"
            image_grid_thw = None

        # Shape description:
        #
        # pixel_values_[videos/images]: (BATCH_SIZE, N_PATCH, HIDDEN_SIZE)
        # [video/image]_grid_thw: (BATCH_SIZE, 3)
        # second_per_grid_ts: (BATCH_SIZE, 1)
        # pixel_values_[videos/images]_lengths_per_sample: (BATCH_SIZE, 1)
        batch = {}
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos

        if video_grid_thw is not None:
            batch["video_grid_thw"] = video_grid_thw

        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw

        # Pad the input_ids, logprob_masks
        batch["input_ids"] = torch.tensor(
            [
                x["input_ids"][:computed_max_len]
                + [self.tokenizer.pad_token_id]
                * (max(0, computed_max_len - len(x["input_ids"])))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        if "label_ids" in processed_samples[0]:
            batch["label_ids"] = torch.tensor(
                [
                    x["label_ids"][:computed_max_len]
                    + [IGNORE_LABEL_ID]
                    * (max(0, computed_max_len - len(x["label_ids"])))
                    for x in processed_samples
                ],
                dtype=torch.long,
            )
        batch["logprob_masks"] = torch.tensor(
            [
                x["logprob_masks"][:computed_max_len]
                + [0] * (max(0, computed_max_len - len(x["logprob_masks"])))
                for x in processed_samples
            ],
            dtype=torch.bool,
        )

        assert len(batch["input_ids"]) == len(
            batch["logprob_masks"]
        ), "The length of input_ids, logprob_masks should be the same"

        padded_tensors = []
        for sample in processed_samples:
            pad_length = computed_max_len - sample["position_ids"].shape[2]
            padded_tensor = torch.nn.functional.pad(
                sample["position_ids"], (0, pad_length), "constant", 1
            )
            padded_tensors.append(padded_tensor)
        batch["position_ids"] = torch.cat(padded_tensors, dim=1)

        return batch

    def get_policy_input(
        self,
        sample: "Qwen3_VL_DataPacker.Payload",
        rollout_output: Optional[str] = None,
        n_ignore_prefix_tokens: int = 0,
        add_generation_prompt: bool = True,
    ) -> Any:
        # assert all(
        #     isinstance(x, dict) and "role" in x and "content" in x for x in sample
        # ), "All samples should be in conversation format, but got: {}".format(sample)
        x = self._process_single_sample(
            sample,
            add_generation_prompt=add_generation_prompt,
        )

        return_dict = {
            "position_ids": x["position_ids"],
        }
        if "pixel_values_videos" in x:
            return_dict["pixel_values_videos"] = x["pixel_values_videos"]
        else:
            return_dict["pixel_values_videos"] = None

        if "video_grid_thw" in x:
            return_dict["video_grid_thw"] = x["video_grid_thw"]
        else:
            return_dict["video_grid_thw"] = None

        if "pixel_values" in x:
            return_dict["pixel_values"] = x["pixel_values"]
        else:
            return_dict["pixel_values"] = None

        if "image_grid_thw" in x:
            return_dict["image_grid_thw"] = x["image_grid_thw"]
        else:
            return_dict["image_grid_thw"] = None

        # Common fields
        input_ids = x["input_ids"]
        completion_ids = []
        if rollout_output:
            completion_ids = self.tokenizer(rollout_output).input_ids  # Don't pad yet
            # recompute position_ids
            # position_ids: (3, 1, seq_len)
            position_ids, _ = self._get_rope_index(
                input_ids=torch.tensor(input_ids + completion_ids).unsqueeze(0).clone(),
                image_grid_thw=torch.tensor(x.get("image_grid_thw"))
                if "image_grid_thw" in x
                else None,
                video_grid_thw=torch.tensor(x.get("video_grid_thw"))
                if "video_grid_thw" in x
                else None,
                attention_mask=None,
            )
            return_dict["position_ids"] = position_ids.clone()
        return_dict["input_ids"] = input_ids + completion_ids

        return_dict["logprob_masks"] = (
            [0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0]
        )

        return_dict["label_ids"] = x["label_ids"]
        return return_dict

    def policy_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        return max([len(x["input_ids"]) for x in processed_samples])

    def policy_collate_fn(
        self, processed_samples: List[Dict[str, Any]], computed_max_len: int
    ) -> Dict[str, Any]:
        for x in processed_samples:
            if "label_ids" in x:
                del x["label_ids"]
        return self._collate_fn(processed_samples, computed_max_len)

    def sft_process_sample(
        self, sample: "Qwen3_VL_DataPacker.Payload"
    ) -> Dict[str, Any]:
        """
        Accepts either raw text or conversation format.
        """
        return self.get_policy_input(sample, add_generation_prompt=False)

    def sft_compute_max_len(self, processed_samples: List[Dict[str, Any]]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        return max([len(x["input_ids"]) for x in processed_samples])

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        # Reuse the RL collate minibatch function
        model_inputs: Dict[str, Any] = self._collate_fn(
            processed_samples, computed_max_len
        )
        del model_inputs["logprob_masks"]
        # Mask the loss on vision padding tokens
        if self.vision_ids is not None:
            assert isinstance(self.vision_ids, list)
            for vision_id in self.vision_ids:
                if vision_id is not None:
                    model_inputs["label_ids"][
                        model_inputs["label_ids"] == vision_id
                    ] = ignore_label_id

        return model_inputs
