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
import copy
from torch.utils.data import Dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import AutoTokenizer
import json

# Each demo video is about 2-4 min,
FPS = 0.05
MAX_PIXELS = 81920
# Repeat the dataset for 10000 times since it only contains 3 samples
FAKE_EPOCH = 10000
# Corresponding to `https://gitlab-master.nvidia.com/tao-fm-applied-research/qwen2.5-vl/-/tree/main/qwen-vl-finetune/demo`
VIDEO_PATH = "/root/cosmos-rl/qwen2.5-vl/qwen-vl-finetune/demo"


class CosmosSFTDataset(Dataset):
    def __init__(self):
        pass

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        self.config = config
        self.tokenizer = tokenizer
        self.video_json = os.path.join(VIDEO_PATH, "video.json")
        self.video_dir = os.path.join(VIDEO_PATH, "videos")
        with open(self.video_json, "r") as file:
            self.dataset = json.load(file)

    def __len__(self):
        return len(self.dataset) * FAKE_EPOCH

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Return a tuple of (prompt, reference answer)
        """
        payload = self.dataset[idx % len(self.dataset)]
        associated_video_files = payload.get("video")

        # Check if the video files exist
        if associated_video_files is not None:
            if isinstance(associated_video_files, str):
                associated_video_files = [associated_video_files]
            else:
                assert isinstance(
                    associated_video_files, list
                ), "Video files must be a list"
            associated_video_files = [
                os.path.join(self.video_dir, video_file)
                for video_file in associated_video_files
            ]
            for video_file in associated_video_files:
                assert os.path.exists(
                    video_file
                ), f"Video file {video_file} does not exist"

        conversations = copy.deepcopy(payload["conversations"])
        content = []

        for conv in conversations:
            conv["role"] = conv.pop("from")
            conv["content"] = conv.pop("value")

            # Normalize role
            conv["role"] = {
                "user": "user",
                "assistant": "assistant",
                "system": "system",
                # for demo dataset
                "gpt": "assistant",
                "human": "user",
            }[conv["role"]]

            if conv["role"] == "user":
                assert isinstance(conv["content"], str), "User message must be string"
                n_video = conv["content"].count("<video>")
                local_conv = []
                for i in range(n_video):
                    local_conv.append(
                        {
                            "type": "video",
                            "video": associated_video_files[i],
                            "max_pixels": MAX_PIXELS,
                            "fps": FPS,
                        }
                    )

                user_content = conv["content"].replace("<video>", "").strip()
                local_conv.append({"type": "text", "text": user_content})
                content.append({"role": conv["role"], "content": local_conv})
            else:
                content.append({"role": conv["role"], "content": conv["content"]})
        return content


if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        return CosmosSFTDataset()

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
    )
