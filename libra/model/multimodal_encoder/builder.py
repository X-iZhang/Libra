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

import os
from .clip_encoder import CLIPVisionTower
from .dino_encoder import DINOVisionTower
from .siglip_encoder import SigLIPVisionTower
from .open_clip_encoder import OpenCLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    vision_tower_config = getattr(vision_tower_cfg, 'mm_vision_tower_config', getattr(vision_tower_cfg, 'vision_tower_config', None))
    vision_tower_checkpoint = getattr(vision_tower_cfg, 'mm_vision_tower_checkpoint', getattr(vision_tower_cfg, 'vision_tower_checkpoint', None))
    
    if vision_tower is None:
        raise ValueError("No vision tower specified in configuration.")

    is_absolute_path_exists = os.path.exists(vision_tower)

    if is_absolute_path_exists or vision_tower.startswith("openai") or \
       vision_tower.startswith("facebook") or vision_tower.startswith("microsoft") or vision_tower.startswith("google"):
        
        if "clip" in vision_tower.lower():
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif "dino" in vision_tower.lower():
            return DINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif "siglip" in vision_tower.lower():
            return SigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            raise ValueError(f'Unknown vision model type in vision_tower: {vision_tower}')

    elif vision_tower.startswith("hf-hub:") or vision_tower_config and vision_tower_checkpoint:
        return OpenCLIPVisionTower(
            vision_tower, args=vision_tower_cfg, vision_tower_config=vision_tower_config, vision_tower_checkpoint=vision_tower_checkpoint, **kwargs
        )
        
    raise ValueError(f'Unknown vision tower: {vision_tower}')