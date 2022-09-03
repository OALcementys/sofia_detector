import torch
import torch.nn as nn

from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import torch.nn.functional as F
import utils
from encoder.vit_adapter import ViTAdapter
#from decoder.fpn import PanopticFPN
#from decoder.fpn_head import FPN
import os


# DecoderLinear(n_cls, patch_size, d_encoder)

# MaskTransformer(n_cls, patch_size, d_encoder,n_layers, n_heads,d_model,d_ff,drop_path_rate, dropout,)

# VisionTransformer(image_size, patch_size, n_layers, d_model, d_ff, n_heads, n_cls, dropout=0.1,
#drop_path_rate=0.0, distilled=False,channels=3,)

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

            #print('pretrained=', state_dict.keys())
            #print('model sate dict=', model.state_dict().keys())

            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # remove `model.` prefix induced by saving models
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            # in case different inp size: change pos_embed
            if "pos_embed" in state_dict.keys():
                pretrained_shape = state_dict['pos_embed'].size()[1]
                model_shape = model.state_dict()['pos_embed'].size()[1]
                if pretrained_shape != model_shape:
                    pos_embed = state_dict['pos_embed']
                    pos_embed = pos_embed.permute(0, 2, 1)
                    pos_embed = F.interpolate(pos_embed, size=model_shape)
                    pos_embed = pos_embed.permute(0, 2, 1)
                    state_dict['pos_embed'] = pos_embed

            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

        else:
            print('wrong checkpoint key')
            print("There is no reference weights available for this model => We use random weights.")
    else:
        print('file {0} does not exist'.format(pretrained_weights))

def build_encoder( pretrained_weights='', key='',trainable=False, arch=None, patch_size=8 , avgpool=False, image_size=224, drop_rate=0):

    arch_dic = {'vit_tiny':{ 'd_model':384, 'n_heads':3, 'n_layers':12},
      'vit_small':{ 'd_model':384, 'n_heads':6, 'n_layers':12},
      'vit_base':{'d_model':384, 'n_heads':12, 'n_layers':12},
      'vit_large':{ 'd_model':384, 'n_heads':24, 'n_layers':12},}

    if arch in arch_dic.keys():
        d_model = arch_dic[arch]['d_model']
        n_heads = arch_dic[arch]['n_heads']
        n_layers = arch_dic[arch]['n_layers']
        d_ff = 4 * d_model
        n_cls = 1 #don't care only usefull for classification head
        # image_size=

        model = ViTAdapter(pretrain_size=image_size, num_heads=n_heads , conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0.,
                 patch_size= patch_size , depth= n_layers, embed_dim= d_model, drop_rate=drop_rate,
                 #image_size=[image_size, image_size, 3], patch_size=patch_size, n_layers=n_layers, embed_dim=d_model,

                 interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                 with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False)

    else:
        print(f"Unknow architecture: {arch}")
        sys.exit(1)

    # load weights to evaluate
    if len(key)>0 and len(pretrained_weights)>0:
        print('found key:', key)
        load_pretrained_weights(model, pretrained_weights, key, arch)
        print('pretrained weights loaded')

    # freeze or not weights
    ct, cf =0 ,0
    if trainable:
        print('trainable encoder', key)
        for p in model.parameters():
            p.requires_grad = True
            ct+= p.numel()
    else:
        print('frozen encoder ', key)
        for p in model.parameters():
            p.requires_grad = False
            cf+= p.numel()
    print(f"{key} adapter built. {ct} trainable params, {cf} frozen params.")
    return model


def build_decoder(pretrained_weights, key, trainable=False, num_cls=2, embed_dim=384, image_size=224, activation=None, smooth=False, add_features=False):
    #dff= mlp_dim
    #model = FPN( num_cls=num_cls, embed_dim=embed_dim, h=image_size , w=image_size, activation=activation, smooth=smooth )
    model = FPN( num_cls=num_cls, embed_dim=embed_dim, h=image_size ,
        w=image_size, activation=activation, smooth=smooth, add_features=add_features )

    # load weights to evaluate
    if len(key)>0 and model!=None:
        arch=""
        load_pretrained_weights(model, pretrained_weights, key, arch)
        print('pretrained weights loaded')


    # freeze or not weights
    ct, cf =0 ,0
    if trainable:
        print('trainable decoder', key)
        for p in model.parameters():
            p.requires_grad = True
            ct+= p.numel()
    else:
        print('frozen decoder ', key)
        for p in model.parameters():
            p.requires_grad = False
            cf+= p.numel()
    print(f"{key} decoder built. {ct} trainable params, {cf} frozen params.")
    return model


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        trainable_encoder,
        trainable_decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trainable_encoder, self.trainable_decoder = trainable_encoder, trainable_decoder


    def forward(self, inp):
        ## encoder
        # [f1, f2, f3, f4] = features
        # f1 (B, D, H//4, W//4)
        # f2 (B, D, H//8, W//8)
        # f3 (B, D, H//16, W//16)
        # f4 (B, D, H//32, W//32)
        if self.trainable_encoder:
            features = self.encoder.forward(inp)
        else:
            with torch.no_grad():
                features = self.encoder.forward(inp)

        ## decoder
        # masks = [B, C, H, W]
        if self.trainable_decoder:
            masks = self.decoder.forward(features )
        else:
            with torch.no_grad():
                masks = self.decoder.forward(features)
        return  masks
        #return features
