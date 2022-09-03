# Copyright (c) Shanghai AI Lab. All rights reserved.
#import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#from mmseg.models.builder import BACKBONES
#from ops.modules import MSDeformAttn
#from ms_deform_attn import MSDeformAttn
from encoder.msda import DeformableHeadAttention
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from functools import partial

from encoder.vit_base import TIMMVisionTransformer
#from vit import VisionTransformer
from encoder.adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

#_logger = logging.getLogger(__name__)


#@BACKBONES.register_module()
class ViTAdapter(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=6, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0.,
                 patch_size= 8 , depth= 12, embed_dim= 384, drop_rate=0,
                 #image_size=[224,224,3], patch_size=8, n_layers=12, embed_dim=384,

                 interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], # 12 layers split in 4 blocks
                 with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False, *args, **kwargs):
    	# VisionTransformer
        # image_size,patch_size,n_layers, d_model, d_ff,n_heads,  n_cls,dropout=0.1,
        #drop_path_rate=0.0, distilled=False, channels=3,

        # TIMMVIsionTransformer
        #img_size=224, patch_size=16, in_chans=3, num_classes=1000,
        #embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
        #drop_rate=0., attn_drop_rate=0., drop_path_rate=0., layer_scale=True,
        #embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #act_layer=nn.GELU, window_attn=False, window_size=14, pretrained=None,
        #with_cp=False
        super().__init__(img_size = pretrain_size, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth,
            drop_rate=drop_rate, norm_layer=nn.LayerNorm, pretrained=pretrained, with_cp=with_cp, *args, **kwargs)

        #super().__init__(image_size=image_size, patch_size=patch_size ,n_layers=n_layers,
        #d_model=embed_dim, d_ff=embed_dim*4, n_heads=num_heads,  n_cls=1 ,dropout=0.1,
        #drop_path_rate=0.0, distilled=False, channels=3,)

        # self.num_classes = 80
        #self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim =  self.embed_dim #self.d_model #

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=nn.LayerNorm ,#self.norm, #partial(nn.LayerNorm, eps=1e-6),  #self.norm_layer
                              with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio,
                             deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        # orginal code: SyncBatchNorm, ours: GroupNorm to allow Gradient Accumulation
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)
        # adaptation to use Gradient Accum
        #self.norm1 = nn.GroupNorm(embed_dim, embed_dim)
        #self.norm2 = nn.GroupNorm(embed_dim, embed_dim)
        #self.norm3 = nn.GroupNorm(embed_dim, embed_dim)
        #self.norm4 = nn.GroupNorm(embed_dim, embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        # added
        self.patch_size = patch_size #8

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // self.patch_size, self.pretrain_size[1] // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        #if isinstance(m, MSDeformAttn):
        if isinstance(m, DeformableHeadAttention):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        [reference_points, spatial_shapes, level_start_index] = deform_inputs1
        # deform_inputs: [reference_points, spatial_shapes, level_start_index]

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)


        # Patch Embedding forward
        x, H, W = self.patch_embed(x)

        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W) #get pos embed from Vit base
        x = self.pos_drop(x + pos_embed) #self.dropout(x + pos_embed) #

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        #original code
        #c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        #c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        #c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        # adaptation: paper: patch_size = 16, ours: patch_size = 8
        c2 = c2.transpose(1, 2).view(bs, dim, H , W ).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H//2 , W//2 ).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 4, W // 4).contiguous()

        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            #original code: paper: patch_size=16,
            #x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            #x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            #x3: same dim
            #x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)

            # adaptation: ours: patch_size=8
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
            #x2 = same dim
            x3 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.25, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
