#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os

import numpy as np
import torch
import torchvision

import yaml
from ../data_helper import LabeledDataset
from ../helper import collate_fn
from integrated_model import BoundingBoxesNN, RoadMapNN


def train(task, dataset, batch_size=2, max_training_scene=25, epochs=0, learning_rate=1e-3,
          save_checkpoints_epochs=1, keep_checkpoint_max=-1, model_dir='./model', device='cpu'):
    

    os.makedirs(model_dir, exist_ok=True)

    logger.info('Loading dataset...')
    image_folder, annotation_csv = dataset

    transform = torchvision.transforms.ToTensor()

    labeled_scene_index_train = np.arange(106, 130)[:max_training_scene]
    labeled_scene_index_valid = np.arange(130, 131)

    labeled_trainset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index_train, transform=transform,
                                      extra_info=False)

    train_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0, collate_fn=collate_fn,
                                               drop_last=True)

    labeled_testset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,
                                     scene_index=labeled_scene_index_valid, transform=transform,
                                     extra_info=False)

    test_loader = torch.utils.data.DataLoader(labeled_testset, batch_size=1, shuffle=True,
                                              num_workers=0, collate_fn=collate_fn)

    if task == 'box':
        nn = BoundingBoxesNN(device=device)
        nn.train(train_loader, test_loader, batch_size=batch_size, epochs=epochs,
                 learning_rate=learning_rate, save_checkpoints_epochs=save_checkpoints_epochs,
                 model_dir=model_dir)

    elif task == 'map':
        nn = RoadMapNN(device=device)
        nn.train(train_loader, test_loader, batch_size=batch_size, epochs=epochs,
                 learning_rate=learning_rate, save_checkpoints_epochs=save_checkpoints_epochs,
                 model_dir=model_dir)


def test(task, dataset, model_path, batch_size=2, verbose=False, device='cpu'):
    # Load data
    image_folder, annotation_csv = dataset

    transform = torchvision.transforms.ToTensor()
    labeled_scene_index_test = np.arange(131, 134)

    labeled_testset = LabeledDataset(image_folder=image_folder, annotation_file=annotation_csv,
                                     scene_index=labeled_scene_index_test, transform=transform,
                                     extra_info=False)

    test_loader = torch.utils.data.DataLoader(labeled_testset, batch_size=batch_size, shuffle=False,
                                              num_workers=0, collate_fn=collate_fn)

    if task == 'box':
        # Initialize model and load parameters
        nn = BoundingBoxesNN(device=device)
        nn.load_model(model_path)
        nn.test(test_loader, batch_size=batch_size, verbose=verbose)

    elif task == 'map':
        # Initialize model and load parameters
        nn = RoadMapNN(device=device)
        nn.load_model(model_path)
        nn.test(test_loader, batch_size=batch_size, verbose=verbose)


if __name__ == '__main__':
    # Process arguments
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-config', '--config', type=str, required=True, help='string argument')
    parser.add_argument('-mode', '--mode', choices=['train', 'test'], help='integer argument')
    parser.add_argument('-checkpoint_path', '--checkpoint_path', type=str, help='string argument')
    parser.add_argument('-log_level', '--log_level', type=int, choices=[0, 10, 20, 30, 40, 50],
                        default=20, help='integer argument')
    parser.add_argument('-verbose', '--verbose', action='store_true')
    args = parser.parse_args()

    # Set logger
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load a configuration file
    with open(args.config, 'r') as yml:
        config = yaml.safe_load(yml)

    dataset = (config['data']['image_folder'], config['data']['annotation_csv'])
    task = config['data']['task']

    if args.mode == 'train':
        # Sample command: `python main.py -config fasterrcnn.config -mode train`
        train(task,
              dataset,
              batch_size=config['train']['batch_size'],
              max_training_scene=config['train']['max_training_scene'],
              epochs=config['train']['epochs'],
              learning_rate=float(config['param']['learning_rate']),
              save_checkpoints_epochs=config['train']['save_checkpoints_epochs'],
              model_dir=config['data']['model_dir'],
              device=device)
    else:
        # Sample command: `python main.py -config fasterrcnn.config -mode test \
        # -checkpoint_path model/boundingBoxNN_model_at_epoch_1.pt`
        test(task,
             dataset,
             args.checkpoint_path,
             batch_size=config['test']['batch_size'],
             verbose=args.verbose,
             device=device)