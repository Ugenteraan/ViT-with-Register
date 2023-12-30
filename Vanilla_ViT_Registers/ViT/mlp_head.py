'''Classification mlp head module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import einops.layers.torch as einops_torch


class MLPHead(nn.Module):
    '''Final classification MLP layer.
    '''

    def __init__(self, patch_embedding_dim, register_token_len, num_classes, expansion_factor=2):
        '''Param init.
        '''
        super(MLPHead, self).__init__()
        self.register_token_len = register_token_len


        self.classification_head = nn.Sequential(einops_torch.Reduce('b n e -> b e', reduction='mean'),
                                                 nn.LayerNorm(patch_embedding_dim),
                                                 nn.Linear(patch_embedding_dim, patch_embedding_dim*expansion_factor),
                                                 nn.GELU(),
                                                 nn.Linear(patch_embedding_dim*expansion_factor, num_classes))


    def forward(self, x):
        '''We'll perform the MLPhead ignoring the register tokens.
        '''
        extracted_cls_token = x[:, 0:1, :] #first position in the patch num dimension. That's where the CLS token was initialized. Should be the size of [batch size, 1,  patch embedding] since we used 0:1 instead of 0 in the first dimension.
        extracted_patch_token = x[:, self.register_token_len+1:, :] #we exclude the cls and register tokens. Only take the patch tokens. 

        #we want to concatenate the tensors along the first dimension. Not the default 0th dimension.
        concat_cls_patch_token = torch.concat([extracted_cls_token, extracted_patch_token], dim=1)

        out = self.classification_head(concat_cls_patch_token)

        return out



