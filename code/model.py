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

from helper import compute_ats_bounding_boxes, compute_ts_road_map

logger = logging.getLogger(__name__)
OUT_SIZE = 800 * 800


class BoundingBoxesNN(object):
    def __init__(self, device='cpu', num_angles=4):
        self.device = device
        self.categories = torch.empty(num_angles + 1).to(self.device)
        for i in range(num_angles + 1):
            self.categories[i] = i * np.pi/(2 * num_angles)

        backbone = models.mobilenet_v2().features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        self.model = FasterRCNN(backbone, num_classes=num_angles + 1,
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
                    t['boxes'] = t.pop('bounding_box').to(self.device)
                    t['labels'] = torch.add(t.pop('category'), 1).to(self.device)
                    xmin = t['boxes'][:, 0].min(dim=1)
                    xmax = t['boxes'][:, 0].max(dim=1)
                    ymin = t['boxes'][:, 1].min(dim=1)
                    ymax = t['boxes'][:, 1].max(dim=1)

                    t['labels'] = self.find_categories(t['boxes'], xmax, ymin)
                    t['boxes'] = torch.stack([torch.mul(torch.add(xmin[0], 40), 918./80.),
                                              torch.mul(torch.add(ymin[0], 40), 512./80.),
                                              torch.mul(torch.add(xmax[0], 40), 918./80.),
                                              torch.mul(torch.add(ymax[0], 40), 512./80.)],
                                             dim=1).type(dtype=torch.float32).to(self.device)
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
            model_path = os.path.join(model_dir, f'4angle_boundingBoxNN_model_at_epoch_{epoch + 1}.pt')
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
            logger.info(f'Validation Bounding Box Score: {score:.5} ({total} samples in total)\n')
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
            logger.info(f'Test Bounding Box Score: {score:.5} ({total} samples in total)\n')

    def predict(self, x):
        preds = []
        for t in x:
            boxes, angles = t['boxes'], t['labels']
            pred_angles = self.get_angles(angles)
            x1s, y1s, x4s, y4s = boxes[:,0], boxes[:,1], boxes[:, 2], boxes[:, 3]
            horiz_side_lens = x4s-x1s
            vert_side_lens = y4s-y1s
            new_x2s, new_y2s = x1s + horiz_side_lens * torch.cos(pred_angles), y1s + horiz_side_lens * torch.sin(pred_angles)
            new_x3s, new_y3s = x1s - vert_side_lens * torch.sin(pred_angles), y1s + vert_side_lens * torch.cos(pred_angles)
            new_x4s, new_y4s = new_x2s - (x1s - new_x3s), new_y2s + (new_y3s - y1s)
            xs = torch.stack([torch.add(torch.mul(x1s, 80./918.), -40),
                              torch.add(torch.mul(new_x3s, 80./918.), -40),
                              torch.add(torch.mul(new_x2s, 80./918.), -40),
                              torch.add(torch.mul(new_x4s, 80./918.), -40)]).t()
            ys = torch.stack([torch.add(torch.mul(y1s, 80./512.), -40),
                              torch.add(torch.mul(new_y3s, 80./512.), -40),
                              torch.add(torch.mul(new_y2s, 80./512.), -40),
                              torch.add(torch.mul(new_y4s, 80./512.), -40)]).t()
            t['boxes'] = torch.stack([xs, ys], dim=1).to(self.device)
            preds.append({'boxes': t['boxes']})
        return preds

    def find_categories(self, boxes, xmax, ymin):
        indices = torch.arange(boxes[:, 0].shape[0]).to(self.device)
        xmax_flat_indices = torch.add(xmax[1], indices * 4)  # KEEP AS 4
        ymin_flat_indices = torch.add(ymin[1], indices * 4)
        x2, y2 = torch.take(boxes[:, 0], xmax_flat_indices), torch.take(boxes[:, 1], xmax_flat_indices)
        x1, y1 = torch.take(boxes[:, 0], ymin_flat_indices), torch.take(boxes[:, 1], ymin_flat_indices)
        angles = torch.atan((y2-y1)/(x2-x1))
        angle_shape = angles.shape[0]
        cat_shape = self.categories.shape[0]
        angle_stack = torch.cat([angles]*self.categories.shape[0]).reshape(self.categories.shape[0], -1).t()
        cat_stack = torch.cat([self.categories]*angle_shape).reshape(-1, cat_shape)
        diff_stack = torch.abs(angle_stack - cat_stack)
        return torch.fmod(diff_stack.min(1)[1], self.categories.shape[0]-1) + 1  # category 0 means no box

    def get_angles(self, angles):
        repeat_cats = torch.cat([self.categories[:-1]]*angles.shape[0])
        indices = torch.arange(angles.shape[0]).to(self.device)
        cat_indices = torch.add(angles-1, indices * self.categories[:-1].shape[0])
        return torch.take(repeat_cats, cat_indices)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        logger.info(f'Loaded model from {model_path}...')

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        logger.info(f'Saved model at {model_path}...')


class ResNet18Baseline(nn.Module):
    def __init__(self, out_size):
        super(ResNet18Baseline, self).__init__()
        self.resnet18 = models.resnet18()
        self.resnet18.fc = nn.Linear(512, out_size)

    def forward(self, x, verbose=False):
        x = self.resnet18(x)
        x = torch.sigmoid(x)
        return x


class RoadMapNN(object):
    def __init__(self, device='cpu'):
        self.device = device
        self.model = ResNet18Baseline(OUT_SIZE).to(self.device)

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
                logger.debug(len(samples))
                samples = torch.stack(samples).to(self.device)
                logger.debug(samples.shape)
                road_images = torch.stack(road_image).type(dtype=torch.float32).to(self.device)
                # (batch_size x 800 x 800) -> (batch size x 640,000)
                road_images = road_images.view(batch_size, -1).to(self.device)

                # Forward
                output = self.model(samples)
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
            logger.info(f'Validation Road Map Score: {score:.5} ({total} samples in total)\n')

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
        logger.info(f'Test Road Map Score: {score:.5} ({total} samples in total)')

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
