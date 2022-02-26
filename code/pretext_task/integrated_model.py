#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

import torch
import torchvision
import torchvision.models as models
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import numpy as np

from torchvision import transforms
from PIL import Image
from ../helper import compute_ats_bounding_boxes, compute_ts_road_map

from jigsaw_model import Network

logger = logging.getLogger(__name__)
OUT_SIZE = 800 * 800
CLASSES = 500

NUM_ANGLES = 4
categories = torch.empty(NUM_ANGLES)
for i in range(NUM_ANGLES):
    categories[i] = i * np.pi/(2*NUM_ANGLES)


def find_category(angle):
    min_diff = np.Inf
    min_angle = 0
    for i,a in enumerate(categories):
        diff = np.absolute(angle - a)
        if (diff < min_diff):
            min_diff = diff
            min_angle = i
    if (np.pi/2 - angle < min_diff):
        min_angle = 0
    return min_angle + 1    # category 0 means no box


def get_angle(category):
    return categories[category-1]


class BoundingBoxesNN(object):
    def __init__(self, device='cpu'):
        self.device = device
        backbone = models.mobilenet_v2().features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        self.model = FasterRCNN(backbone, num_classes=NUM_ANGLES + 1,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler).to(self.device)

    def train(self, train_data, valid_data, batch_size=2, epochs=0, learning_rate=1e-3,
              save_checkpoints_epochs=1, model_dir='./model'):

        # Configure the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            logger.info(f'Starting epoch {epoch + 1}...')
            for batch_idx, (sample, target, road_image) in enumerate(train_data):
                # Prepare data
                samples = [torchvision.utils.make_grid(s, nrow=3, padding=0) for s in sample]
                logger.debug(len(samples))
                samples = torch.stack(samples).to(self.device)
                logger.debug(samples.shape)

                # change dictionary names to those that FasterRCNN takes in, and make boxes have \
                # only bottom left corner, top right corner
                targets = []
                for t in target:
                    t['boxes'] = t.pop('bounding_box')
                    t['labels'] = torch.add(t.pop('category'), 1).to(self.device)
                    xmin = torch.mul(torch.add(t['boxes'][:, 0].min(dim=1)[0], 40), 918./80.)
                    xmax = torch.mul(torch.add(t['boxes'][:, 0].max(dim=1)[0], 40), 918./80.)
                    ymin = torch.mul(torch.add(t['boxes'][:, 1].min(dim=1)[0], 40), 512./80.)
                    ymax = torch.mul(torch.add(t['boxes'][:, 1].max(dim=1)[0], 40), 512./80.)
                    xmin_indices = t['boxes'][:, 0].min(dim=1)[1]
                    xmax_indices = t['boxes'][:, 0].max(dim=1)[1]
                    ymin_indices = t['boxes'][:, 1].min(dim=1)[1]
                    ymax_indices = t['boxes'][:, 1].max(dim=1)[1]

                    for j in range(t['boxes'].shape[0]):
                        t['labels'][j] = find_category(np.arctan((t['boxes'][j][1][xmax_indices[j]] - t['boxes'][j][1][ymin_indices[j]])/
                                                                 (t['boxes'][j][0][xmax_indices[j]] - t['boxes'][j][0][ymin_indices[j]])))
                    t['boxes'] = torch.stack([xmin, ymin, xmax, ymax], dim=1).type(
                                             dtype=torch.float32).to(self.device)
                    targets.append(t)

                # Forward
                loss_dict = self.model(samples, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_val = losses.item()

                # Backward
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Intermediate log
                if batch_idx % 32 == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(sample), len(train_data.dataset),
                        100. * batch_idx / len(train_data), loss_val))
            # Epoch log
            logger.info(f'epoch [{epoch + 1}/{epochs}], loss:{loss_val:.4f}')

            # Save model
            model_path = os.path.join(model_dir, f'boundingBoxNN_model_at_epoch_{epoch + 1}.pt')
            if (epoch + 1) % save_checkpoints_epochs == 0:
                self.save_model(model_path)

            self.validate(valid_data)
        logger.info(f'Training done ({epochs} epochs)')

    def validate(self, valid_data):
        # Evaluate on the validation data
        self.model.eval()

        predictions, golds = [], []
        with torch.no_grad():
            for batch_idx, (sample, target, road_image) in enumerate(valid_data):
                samples = [torchvision.utils.make_grid(s, nrow=3, padding=0) for s in sample]
                logger.debug(len(samples))
                samples = torch.stack(samples).to(self.device)
                logger.debug(samples.shape)
                output = self.model(samples)
                logger.debug(output)
                prediction = self.predict(output)
                logger.debug(prediction)
                predictions.append(prediction)
                golds.append(target)

            total = 0
            total_ats_bounding_boxes = 0
            for i, (pred, gold) in enumerate(zip(predictions, golds)):
                for j in range(len(pred)):
                    total += 1
                    ats_bounding_boxes = compute_ats_bounding_boxes(
                                            pred[j]['boxes'].to(self.device),
                                            gold[j]['bounding_box'].to(self.device))
                    total_ats_bounding_boxes += ats_bounding_boxes
            score = total_ats_bounding_boxes / total
            logger.info(f'Validation Bounding Box Score: {score:.4} ({total} samples in total)\n')
        self.model.train()

    def test(self, test_data, batch_size=1, verbose=False):
        predictions, golds = [], []
        with torch.no_grad():
            for batch_idx, (sample, target, road_image) in enumerate(test_data):
                samples = [torchvision.utils.make_grid(s, nrow=3, padding=0) for s in sample]
                logger.debug(len(samples))
                samples = torch.stack(samples).to(self.device)
                logger.debug(samples.shape)
                output = self.model(samples)
                prediction = self.predict(output)
                predictions.append(prediction)
                golds.append(target)

            total = 0
            total_ats_bounding_boxes = 0
            for i, (pred, gold) in enumerate(zip(predictions, golds)):
                for j in range(len(pred)):
                    total += 1
                    ats_bounding_boxes = compute_ats_bounding_boxes(
                                            pred[j]['boxes'].to(self.device),
                                            gold[j]['bounding_box'].to(self.device))
                    total_ats_bounding_boxes += ats_bounding_boxes
            score = total_ats_bounding_boxes / total
            logger.info(f'Validation Bounding Box Score: {score:.4} ({total} samples in total)\n')

    def predict(self, x):
        preds = []
        for t in x:
            boxes, angles = t['boxes'], t['labels']
            pred_boxes = []
            for box, angle in zip(boxes, angles):
                x1, y1 = box[0],box[1]
                pred_angle = get_angle(angle)
                x4, y4 = box[2],box[3]
                horiz_side_len = x4-x1
                vert_side_len = y4-y1
        
                x2, y2 = x1 + horiz_side_len * np.cos(pred_angle), y1 + horiz_side_len * np.sin(pred_angle)
                x3, y3 = x1 - vert_side_len * np.sin(pred_angle), y1 + vert_side_len * np.cos(pred_angle)
                x4, y4 = x2 - (x1-x3), y2 + (y3-y1)
                pred_boxes.append(torch.tensor([[torch.add(torch.mul(x1, 80./918.), -40), torch.add(torch.mul(x3, 80./918.), -40), torch.add(torch.mul(x2, 80./918.), -40), torch.add(torch.mul(x4, 80./918.), -40)],[torch.add(torch.mul(y1, 80./512.), -40), torch.add(torch.mul(y3, 80./512.), -40), torch.add(torch.mul(y2, 80./512.), -40), torch.add(torch.mul(y4, 80./512.), -40)]]))
            if (pred_boxes != []):
                logger.debug('not empty')
                pred_boxes = torch.stack(pred_boxes).type(dtype=torch.float32).to(self.device)
            else:
                logger.debug('empty')                
            preds.append({'boxes': pred_boxes})
        return preds

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        logger.info(f'Loaded model from {model_path}...')

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        logger.info(f'Saved model at {model_path}...')

def get_transform_jigsaw(image):
    if np.random.rand() < 0.30:
        image = image.convert('LA').convert('RGB')
    
    if image.size[0] != 255:
        image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        image = image_transformer(image)

    s = float(image.size[0]) / 3
    a = s / 2
    tiles = [None] * 9
    for n in range(9):
        i = n / 3
        j = n % 3
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
        tile = image.crop(c.tolist())

        def rgb_jittering(im):
            im = np.array(im, 'int32')
            for ch in range(3):
                im[:, :, ch] += np.random.randint(-2, 2)
            im[im > 255] = 255
            im[im < 0] = 0
            return im.astype('uint8')

        augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])
        tile = augment_tile(tile)

        # Normalize the patches indipendently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
        tile = norm(tile)
        tiles[n] = tile

    all_perm = np.load('permutations_hamming_max_%d.npy' % (CLASSES))
    if all_perm.min() == 1:
        all_perm = all_perm - 1

    order = np.random.randint(len(all_perm))
    data = [tiles[all_perm[order][t]] for t in range(9)]
    data = torch.stack(data, 0)
    #print("data shape", data.shape)
    return data, int(order), tiles

class ResNet18Baseline(nn.Module):
    def __init__(self, out_size):
        super(ResNet18Baseline, self).__init__()
        # (batch_size x 6 x 4096) -> (batch_size x 3 x 800 x 800)
        self.fc1 = nn.Linear(6*4096, 800*800)
        self.resnet18 = models.resnet18()
        self.resnet18.fc = nn.Linear(512, out_size)

    def forward(self, x, verbose=False):
        # (batch_size x 4096*6) -> (batch_size x 3 x 800 x 800)
        x = torch.reshape(x, (-1, 6*4096))
        x = self.fc1(x)
        x = torch.reshape(x, (-1, 800, 800))
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet18(x)
        x = torch.sigmoid(x)
        return x


class RoadMapNN(object):
    def __init__(self, device='cpu'):
        self.device = device
        self.model = ResNet18Baseline(OUT_SIZE).to(self.device)
        self.pretask_model = Network().to(self.device)
        self.pretask_model.load('checkpts_jigsaw/1024/100_jps_001.pth')
        self.pretask_model.eval()

    def train(self, train_data, valid_data, batch_size=2, epochs=0, learning_rate=1e-3,
              save_checkpoints_epochs=1, model_dir='./model'):

        # Define loss function
        criterion = nn.MSELoss()

        # Configure the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            logger.info(f'Starting epoch {epoch + 1}...')
            for batch_idx, (sample, target, road_image) in enumerate(train_data):
                # Prepare data
                samples = [torchvision.utils.make_grid(s, nrow=3, padding=0) for s in sample]
                _samples = []
                for s in sample:
                    feature_vectors = []
                    for image in s:
                        image = transforms.ToPILImage()(image).convert("RGB")
                        data, _, __ = get_transform_jigsaw(image)
                        data = data.unsqueeze(0)
                        data = data.to(self.device)
                        feature_vector = self.pretask_model(data)
                        feature_vectors.append(feature_vector)
                    feature_vectors = torch.stack(feature_vectors)
                    _samples.append(feature_vectors)
                
                _samples = torch.stack(_samples)
                _samples = torch.squeeze(_samples) 

                road_images = torch.stack(road_image).type(dtype=torch.float32).to(self.device)
                # (batch_size x 800 x 800) -> (batch size x 640,000)
                road_images = road_images.view(batch_size, -1).to(self.device)

                # Forward
                output = self.model(_samples)
                loss = criterion(output, road_images)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Intermediate log
                if batch_idx % 32 == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(sample), len(train_data.dataset),
                        100. * batch_idx / len(train_data), loss.item()))
            # Epoch log
            logger.info(f'epoch [{epoch + 1}/{epochs}], loss:{loss.item():.4f}')

            # Save model
            model_path = os.path.join(model_dir, f'roadMapNN_model_at_epoch_{epoch + 1}.pt')
            if (epoch + 1) % save_checkpoints_epochs == 0:
                self.save_model(model_path)

            # Validate
            self.validate(valid_data)
        logger.info(f'Training done ({epochs} epochs)')

    def validate(self, valid_data):
        # Evaluate on the validation data with the saved model
        logger.info('Running validation...')
        self.model.eval()

        predictions, golds = [], []
        with torch.no_grad():
            for batch_idx, (sample, target, road_image) in enumerate(valid_data):
                samples = [torchvision.utils.make_grid(s, nrow=3, padding=0) for s in sample]
                logger.debug(len(samples))
                samples = torch.stack(samples).to(self.device)
                logger.debug(samples.shape)
                output = self.model(samples)
                prediction = self.predict(output)
                predictions.append(prediction)
                road_images = torch.stack(road_image).to(self.device)
                golds.append(road_images)
            golds = torch.cat(golds, dim=0)
            predictions = torch.cat(predictions, dim=0)  # batch size x 800 x 800

            total = 0
            total_ts_road_map = 0
            for i, (pred, gold) in enumerate(zip(predictions, golds)):
                total += 1
                ts_road_map = compute_ts_road_map(pred, gold)
                total_ts_road_map += ts_road_map
            score = total_ts_road_map / total
            logger.info(f'Validation Road Map Score: {score:.4} ({total} samples in total)\n')

        self.model.train()

    def test(self, test_data, batch_size=1, verbose=False):
        logger.info('Running test...')
        predictions, golds = [], []
        with torch.no_grad():
            for batch_idx, (sample, target, road_image) in enumerate(test_data):
                samples = [torchvision.utils.make_grid(s, nrow=3, padding=0) for s in sample]
                logger.debug(len(samples))
                samples = torch.stack(samples).to(self.device)
                logger.debug(samples.shape)
                output = self.model(samples)
                prediction = self.predict(output)
                predictions.append(prediction)
                road_images = torch.stack(road_image).to(self.device)
                golds.append(road_images)
            golds = torch.cat(golds, dim=0)
            predictions = torch.cat(predictions, dim=0)  # batch size x 800 x 800

            total = 0
            total_ts_road_map = 0
            for i, (pred, gold) in enumerate(zip(predictions, golds)):
                total += 1
                ts_road_map = compute_ts_road_map(pred, gold)
                total_ts_road_map += ts_road_map
                if verbose:
                    logger.info(f'{i} - Road Map Score: {ts_road_map:.4}')
            score = total_ts_road_map / total
        logger.info(f'Test Road Map Score: {score:.4} ({total} samples in total)')

    def predict(self, x):
        x = x > 0.5
        # (batch_size x 640,000) -> (batch_size x 800 x 800)
        x = x.view(-1, 800, 800)
        return x

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        logger.info(f'Loaded model from {model_path}...')

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        logger.info(f'Saved model at {model_path}...')