#    Copyright 2025 Xi Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import os
import torch

from timm.models.vision_transformer import VisionTransformer
from transformers.modeling_outputs import BaseModelOutput

from .utils import from_pretrained, remove_transformer_pooler_weights

libra_LLAVARAD_HF_REPO = "X-iZhang/libra-llava-rad"

class VisionTower(torch.nn.Module):

    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit
        self.hidden_size = vit.embed_dim
        self.num_patches = vit.patch_embed.num_patches

    @torch.no_grad()
    def forward(self, images, output_hidden_states=True):
        hidden_states = self.vit.patch_embed(images)
        hidden_states = self.vit._pos_embed(hidden_states)
        hidden_states = self.vit.norm_pre(hidden_states)
        block_states = [hidden_states]
        for block in self.vit.blocks:
            hidden_states = block(hidden_states)
            block_states.append(hidden_states)
        if output_hidden_states:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=block_states
            )
        else:
            return BaseModelOutput(last_hidden_state=hidden_states)

class Processor:
    def __init__(self, fn) -> None:
        self.fn = fn  

    def preprocess(self, image, return_tensors="pt"):
        if return_tensors != "pt":
            raise NotImplementedError

        if isinstance(image, list):
            tensor_list = [self.fn(im) for im in image]
            batched = torch.stack(tensor_list) 
        else:
            batched = self.fn(image).unsqueeze(0)

        return {"pixel_values": batched}

    def __call__(self, image, return_tensors="pt"):
        return self.preprocess(image, return_tensors=return_tensors)

class OpenCLIPVisionTower(torch.nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, vision_tower_config=None, vision_tower_checkpoint=None):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        if os.path.exists(vision_tower_config):
            self.vision_tower_config = json.load(open(vision_tower_config))
        else:
            # likely from hf hub
            from huggingface_hub import hf_hub_download
            cache_file = hf_hub_download(repo_id=libra_LLAVARAD_HF_REPO, filename='biomedclipcxr_518.json')
            self.vision_tower_config = json.load(open(cache_file))
            
        self.vision_tower_checkpoint = vision_tower_checkpoint
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = vision_tower
        
    def load_model(self):
        if self.vision_tower_checkpoint:
            if not os.path.exists(self.vision_tower_checkpoint):
 
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id=libra_LLAVARAD_HF_REPO, filename="", subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return cache_file
                self.vision_tower_checkpoint = load_from_hf(filename=self.vision_tower_checkpoint)

            self.vision_tower_checkpoint = remove_transformer_pooler_weights(self.vision_tower_checkpoint)
        model, preprocess, _ = from_pretrained(
            self.vision_tower_name, self.vision_tower_config, self.vision_tower_checkpoint
        )
        self.image_processor = Processor(preprocess)

        self.vision_tower = VisionTower(model.visual.trunk)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
            
        return torch.stack([image_features])

    @torch.no_grad()
    def forward(self, images):
        if images.shape[0] != 2:
            raise ValueError(
                f"Expected images.shape[0] == 2, but got {images.shape[0]}. "
                "Ensure the input includes both current and previous images."
            )

        cur_images = images[0]  
        prev_images = images[1]  

        cur_image_forward_outs = self.vision_tower(cur_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cur_features = self.feature_select(cur_image_forward_outs)  

        prev_image_forward_outs = self.vision_tower(prev_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        prev_features = self.feature_select(prev_image_forward_outs)  

        cur_features = cur_features.permute(1, 0, 2, 3) 
        prev_features = prev_features.permute(1, 0, 2, 3) 

        # Stack current and previous images along a new dimension
        images_features = torch.stack([cur_features, prev_features])  

        return images_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        raise NotImplementedError

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        return self.vision_tower.num_patches

    