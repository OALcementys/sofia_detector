#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
import argparse
import warnings
warnings.filterwarnings('ignore')
from encoder.adapter_modules import deform_inputs
from encoder.vit_adapter import ViTAdapter
from encoder.adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

from  torchvision.transforms import ToPILImage as T
from models import build_encoder, build_decoder, Segmenter
from model import *
from config import *
from functions import *
# In[3]:

import utils
from torch.autograd import Variable


import sys
sys.path.append('../')

"""
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.init_process_group('gloo',
                                     init_method="env://",
                                     rank=0,
                                     world_size=1)
"""


class ViTAdapter(object):
    """
    weights : encoder weights trainned on segmentation task
    config : config.py class (trainning hyper parameteres)

    """
    def __init__(self,weights,config):


        self.config=config
        self.rpn=RPN(len(self.config.RPN_ANCHOR_RATIOS), self.config.RPN_ANCHOR_STRIDE, 384).cuda()
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                        config.RPN_ANCHOR_RATIOS,
                                                                                        config.BACKBONE_SHAPES,
                                                                                        config.BACKBONE_STRIDES,
                                                                                        config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        self.classifier = Classifier(384, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES).cuda()
        self.encoder=self.load_model(weights=weights)


    def load_encoder(weights=None):

        encoder = build_encoder(pretrained_weights='', key='encoder', trainable=False, arch='vit_small', patch_size=8,  image_size=244)
        #decoder = build_decoder(pretrained_weights='', key='decoder',trainable=False, num_cls=2, embed_dim=encoder.embed_dim, image_size=244,)
        #model = Segmenter(encoder, decoder, False, False)
        return encoder.eval().cuda()



    ### input shape out of dataloader for each image_index should be  [images, image_metas, gt_class_ids, gt_boxes]
    ## images : input images
    ## images_meta :     """Takes attributes of an image and puts them in one 1D array. Use utils.parse_image_meta() to parse the values back
    def predict(self, input, mode):
        molded_images = input[0]
        image_metas = input[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

        ### getting features from encoder
        ### features composition
        #   features=[f1, f2, f3, f4]
        # f1 (B, D, H//4, W//4)
        # f2 (B, D, H//8, W//8)
        # f3 (B, D, H//16, W//16)
        # f4 (B, D, H//32, W//32)

        features=self.model(molded_images)

        ### running RPN on each feature map to get proposition on each
        layer_outputs=[]
        for p in features:
            layer_outputs.append(rpn(p))

        ### concatenate output on a single feature map
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE

        ## region proposal on the feature map
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                 proposal_count=proposal_count,
                                 nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                 anchors=self.anchors,
                                 config=self.config)

        if mode == 'inference': ## validation
                ###  getting class prediction and bounding box from classifier network (1st auxiliary output)
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = classifier(features, rpn_rois)

            ##getting bbox prediction from the detection layer (2nd  auxiliary output)
            detections = detection_layer(config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)
                        h, w = self.config.IMAGE_SHAPE[:2]


            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()

            ##normalizing pred bbox
            detection_boxes = detections[:, :4] / scale

            # Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)


            # Add back batch dimension
            detections = detections.unsqueeze(0)

            return detections

        elif mode == 'training':

            ### input shape [images, image_metas, gt_class_ids, gt_boxes]
            gt_class_ids = input[2]
            gt_boxes = input[3]

            # Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if not rois.size():
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                #mrcnn_mask = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    #mrcnn_mask = mrcnn_mask.cuda()
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                # Create masks for detections
                #mrcnn_mask = self.mask(mrcnn_feature_maps, rois)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,]


    def detect(self, images):



        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

    # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

    # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # Wrap in variable
        molded_images = Variable(molded_images, volatile=True)

        # Run object detection
        detections = self.predict([molded_images, image_metas], mode='inference')

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        ##mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores =\
                utils.unmold_detections(detections[i],                  ### modify unmold_detections to not work on masks
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
            })
        return results


def train_epoch(model, datagenerator, optimizer, steps):
    batch_count = 0
    loss_sum = 0
    loss_rpn_class_sum = 0
    loss_rpn_bbox_sum = 0
    loss_mrcnn_class_sum = 0
    loss_mrcnn_bbox_sum = 0
    loss_mrcnn_mask_sum = 0
    step = 0

    optimizer.zero_grad()

    for inputs in datagenerator:
        batch_count += 1

        images = inputs[0]
        image_metas = inputs[1]
        rpn_match = inputs[2]
        rpn_bbox = inputs[3]
        gt_class_ids = inputs[4]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        # image_metas as numpy array
        image_metas = image_metas.numpy()

        # Wrap in variables
        images = Variable(images)
        rpn_match = Variable(rpn_match)
        rpn_bbox = Variable(rpn_bbox)
        gt_class_ids = Variable(gt_class_ids)
        gt_boxes = Variable(gt_boxes)
        gt_masks = Variable(gt_masks)

        # To GPU
        if self.config.GPU_COUNT:
            images = images.cuda()
            rpn_match = rpn_match.cuda()
            rpn_bbox = rpn_bbox.cuda()
            gt_class_ids = gt_class_ids.cuda()
            gt_boxes = gt_boxes.cuda()
            #gt_masks = gt_masks.cuda()

        # Run object detection
        rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
            self.predict([images, image_metas, gt_class_ids, gt_boxes], mode='training')

        # Compute losses
        rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss#, mrcnn_mask_loss =
        =compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
        loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
        if (batch_count % self.config.BATCH_SIZE) == 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_count = 0

        # Progress
        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                         suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".format(
                             loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                             mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                            # mrcnn_mask_loss.data.cpu()[0]
                             ), length=10)

        # Statistics
        loss_sum += loss.data.cpu()[0]/steps
        loss_rpn_class_sum += rpn_class_loss.data.cpu()[0]/steps
        loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0]/steps
        loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0]/steps
        loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0]/steps
        #loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0]/steps

        # Break after 'steps' steps
        if step==steps-1:
            break
        step += 1

    return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum
    #, loss_mrcnn_mask_sum

def valid_epoch(model,datagenerator, steps):

        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0

        for inputs in datagenerator:
            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            ##gt_masks = inputs[6]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            images = Variable(images, volatile=True)
            rpn_match = Variable(rpn_match, volatile=True)
            rpn_bbox = Variable(rpn_bbox, volatile=True)
            gt_class_ids = Variable(gt_class_ids, volatile=True)
            gt_boxes = Variable(gt_boxes, volatile=True)
            gt_masks = Variable(gt_masks, volatile=True)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask# mrcnn_mask = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            if not target_class_ids.size():
                continue

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss                #, mrcnn_mask_loss =
            = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss          #+ mrcnn_mask_loss

            # Progress
            printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                             suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".format(
                                 loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                 mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                 mrcnn_mask_loss.data.cpu()[0]), length=10)

            # Statistics
            loss_sum += loss.data.cpu()[0]/steps
            loss_rpn_class_sum += rpn_class_loss.data.cpu()[0]/steps
            loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0]/steps
            loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0]/steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0]/steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0]/steps

            # Break after 'steps' steps
            if step==steps-1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum


def train_model( train_dataset, val_dataset, learning_rate, epochs):

        """



        # Data generators
        train_set = Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
        val_set = Dataset(val_dataset, self.config, augment=True)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch+1, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))

        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(1, epochs+1):
            log("Epoch {}/{}.".format(epoch,epochs))

            # Training
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox])
            self.val_loss_history.append([val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox])
            ##visualize.plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch))
