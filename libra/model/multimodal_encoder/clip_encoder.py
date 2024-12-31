#    Copyright 2024 Xi Zhang
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

import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer 
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch') 

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
    
    def get_features(self, images):
        outputs = self.vision_tower(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        if self.select_layer == "all":
            if self.select_feature == "patch":
                all_layers_features = [hidden_state[:, 1:, :].contiguous() for hidden_state in hidden_states[1:]]
            elif self.select_feature == "cls_patch":
                all_layers_features = [hidden_state.contiguous() for hidden_state in hidden_states[1:]]
            else:
                raise ValueError(f"Unexpected select feature: {self.select_feature}")

            return torch.stack(all_layers_features) 
        else:
            selected_layer_features = hidden_states[int(self.select_layer)]

            if self.select_feature == "patch":
                selected_layer_features = selected_layer_features[:, 1:]
            elif self.select_feature == "cls_patch":
                selected_layer_features = selected_layer_features
            else:
                raise ValueError(f"Unexpected select feature: {self.select_feature}")

            return torch.stack([selected_layer_features])
    
    @torch.no_grad()
    def forward(self, images):
        
        if images.shape[0] != 2:
            raise ValueError(
                f"Expected images.shape[0] == 2, but got {images.shape[0]}. "
                "Ensure the input includes both current and previous images."
            )

        cur_images = images[0]  
        prev_images = images[1]  

        cur_features = self.get_features(cur_images)  
        prev_features = self.get_features(prev_images)  
        
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
        
        return self.vision_tower.dtype 

    @property
    def device(self):
        return self.vision_tower.device 

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size 

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_layers(self):
        return self.config.num_hidden_layers 