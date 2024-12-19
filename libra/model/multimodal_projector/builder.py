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
import torchvision.ops as ops
import re


class TAC(nn.Module):
    def __init__(self, config):
        super(TAC,self).__init__()

        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.dropout = 0.1
        self.layers_number = 12 # RAD-DINO hidden layers

        # LFE
        self.LFE = nn.Sequential(
            ops.SqueezeExcitation(self.layers_number,self.layers_number // 2,activation=nn.GELU),
            nn.Conv2d(self.layers_number,self.layers_number // 2,kernel_size=1,bias=False),
            ops.SqueezeExcitation(self.layers_number // 2,self.layers_number // 4,activation=nn.GELU),
            nn.Conv2d(self.layers_number // 2,self.layers_number // 4,kernel_size=1,bias=False),
            ops.SqueezeExcitation(self.layers_number // 4,1,activation=nn.GELU),
            nn.Conv2d(self.layers_number // 4,1,kernel_size=1,bias=False)
        )
  
        self.LFE_prior_bias = nn.Parameter(torch.tensor(0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.LFE_cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        # self-attention
        self.cur_self_attention = nn.MultiheadAttention(embed_dim=(self.mm_hidden_size), num_heads=self.num_attention_heads,batch_first=True,add_bias_kv=True)
        self.prior_self_attention = nn.MultiheadAttention(embed_dim=(self.mm_hidden_size), num_heads=self.num_attention_heads,batch_first=True,add_bias_kv=True)
        self.cros_attention = nn.MultiheadAttention(embed_dim=(self.mm_hidden_size), num_heads=self.num_attention_heads,batch_first=True,add_bias_kv=True)
        
        self.norm1 = nn.LayerNorm(self.mm_hidden_size)
        self.norm2 = nn.LayerNorm(self.mm_hidden_size)
        self.norm3 = nn.LayerNorm(self.mm_hidden_size)
        self.norm4 = nn.LayerNorm(self.mm_hidden_size)
        
        self.mlp_attn = nn.Sequential(
            nn.Linear(self.mm_hidden_size, self.mm_hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mm_hidden_size, self.mm_hidden_size),
            nn.Dropout(self.dropout)
        )

        self.mlp_final = nn.Sequential(
            nn.Linear(self.mm_hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)
    
    def calculate_cosine_similarity(self, tensor1, tensor2):

        assert tensor1.shape == tensor2.shape, "The shapes of the two tensors must be the same"

        tensor1_flat = tensor1.view(tensor1.size(0), -1)
        tensor2_flat = tensor2.view(tensor2.size(0), -1)

        tensor1_flat_normalized = tensor1_flat / tensor1_flat.norm(dim=-1, keepdim=True)
        tensor2_flat_normalized = tensor2_flat / tensor2_flat.norm(dim=-1, keepdim=True)

        cosine_similarities = self.LFE_cos(tensor1_flat_normalized, tensor2_flat_normalized)
        cosine_similarities_normalized = ((cosine_similarities + 1) / 2).pow(8)
        cosine_similarities_normalized = cosine_similarities_normalized.view(-1, 1, 1)

        return cosine_similarities_normalized 
    
    # self-attention block   
    def cur_self_att_block(self,x):
        x = self.cur_self_attention(x,x,x)[0]
        return self.dropout1(x)
    # self-attention block   
    def prior_self_att_block(self,x):
        x = self.prior_self_attention(x,x,x)[0]
        return self.dropout2(x)
    # cross attention block 
    def cros_att_block(self,x,y):
        x = self.cros_attention(x,y,y)[0]
        return self.dropout3(x)
    
    #TFM
    def TFM(self,cur_features,prev_features):
        
        cur_features_temp = cur_features
        prev_features_temp = prev_features
        
        cos= self.calculate_cosine_similarity(cur_features_temp,prev_features_temp)
        prev_weight = cos * self.LFE_prior_bias
        prev_features_temp = prev_features_temp + prev_weight
        
        cur_features = self.norm1(cur_features_temp + self.cur_self_att_block(cur_features_temp))
        prev_features = self.norm2(prev_features_temp + self.prior_self_att_block(prev_features_temp))
        combined_features = self.norm3(cur_features + self.cros_att_block(cur_features,prev_features))

        output = self.norm4(cur_features_temp + self.mlp_attn(combined_features))
        output = self.mlp_final(output)

        return output   
    
    def forward(self, image_features, *args, **kwargs): 
        cur_features, prev_features = image_features
        
        cur_features = self.LFE(cur_features).squeeze(1)
        prev_features= self.LFE(prev_features).squeeze(1)
        
        output = self.TFM(cur_features,prev_features)

        return output    
    
    @property
    def config(self):
        return {"mm_projector_type": 'TAC'}

class Projector(nn.Module):
    def __init__(self, base_projector):
        super().__init__()
        self.projector = base_projector

    def forward(self, image_features, *args, **kwargs):
        temp_features = image_features[0].squeeze(1)
        return self.projector(temp_features)


def build_vision_projector(config, delay_load=False, *args,**kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if projector_type == 'linear':
        linear_layer = nn.Linear(config.mm_hidden_size, config.hidden_size)
        return Projector(linear_layer)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)] 
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return Projector(nn.Sequential(*modules))
    
    if projector_type == 'TAC':
        return TAC(config)

    raise ValueError(f'Unknown projector type: {projector_type}')