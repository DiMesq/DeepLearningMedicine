import os
import glob
import logging
import time
import json
from parse_args import parse_args
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from pcam_dataset import PcamDataset

NUM_CLASSES = 2

def initialize_resnet(model_name, pretrained):
    if model_name == 'resnet18':
        model_fn = models.resnet18
    elif model_name == 'resnet50':
        model_fn = models.resnet50
    elif model_name == 'resnet101':
        model_fn = models.resnet101
    elif model_name == 'resnet152':
        model_fn = models.resnet152

    model = model_fn(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)
    input_size = 224

    return model, input_size


def initialize_densenet(model_name, pretrained):
    if model_name == 'densenet121':
        model_fn = models.densenet121
    elif model_name == 'densenet161':
        model_fn = models.densenet161
    elif model_name == 'densenet201':
        model_fn = models.densenet201

    model = model_fn(pretrained=pretrained)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, NUM_CLASSES)
    input_size = 224

    return model, input_size


def initialize_model(model_name, pretrained=False):
    logging.info(f'Initializing {model_name} model (pretrained? {pretrained}) ...')
    if 'resnet' in model_name:
        model, input_size = initialize_resnet(model_name, pretrained)
    if 'densenet' in model_name:
        model, input_size = initialize_densenet(model_name, pretrained)
    return model, input_size


def get_dataloaders(input_size, batch_size, local, test_run=None, negative_only=False):
    images_path = '/scratch/dam740/DLM/data/images/train'
    train_labels_path = '/scratch/dam740/DLM/data/train_labels.csv'
    val_labels_path = '/scratch/dam740/DLM/data/val_labels.csv'

    if test_run:
        train_labels_path = f"{train_labels_path[:train_labels_path.find('.csv')]}_small.csv"
        val_labels_path = f"{val_labels_path[:val_labels_path.find('.csv')]}_small.csv"

    if local:
        images_path = 'data/local_test'
        train_labels_path = 'data/test_labels.csv'
        val_labels_path = 'data/test_labels.csv'

    train_transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.RandomChoice([
                                 transforms.ColorJitter(brightness=.5),
                                 transforms.ColorJitter(contrast=.5),
                                 transforms.ColorJitter(saturation=.5),
                                 transforms.ColorJitter(hue=.5),
                                 transforms.ColorJitter(.1, .1, .1, .1),
                                 transforms.ColorJitter(.3, .3, .3, .3),
                                 transforms.ColorJitter(.5, .5, .5, .5),
                                ]),
        transforms.RandomChoice([
                                transforms.RandomRotation((0, 0)),
                                transforms.RandomRotation((90, 90)),
                                transforms.RandomRotation((180, 180)),
                                transforms.RandomRotation((270, 270)),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.RandomVerticalFlip(p=1),
                                transforms.Compose([
                                                    transforms.RandomHorizontalFlip(p=1),
                                                    transforms.RandomRotation((90, 90))
                                                    ]),
                                transforms.Compose([
                                                    transforms.RandomHorizontalFlip(p=1),
                                                    transforms.RandomRotation((270, 270))
                                                    ])

                                ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # train data loader
    train_dataset = PcamDataset(images_path, train_labels_path, train_transformations, negative_only)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # val data loader
    val_dataset = PcamDataset(images_path, val_labels_path, val_transformations, negative_only)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader}


def get_model_path(model_name, local, run_id):
    root_dir = 'models' if local else '/scratch/dam740/DLM/models'
    model_path = f'{root_dir}/{model_name}_{run_id}'
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path


def get_run_id(local):
    root_dir = 'models' if local else '/scratch/dam740/DLM/models'
    files = os.listdir(root_dir)
    if files:
        myid = max([int(file.strip().split('_')[-1]) for file in files]) + 1
    else:
        myid = 1
    return str(myid)


def train_loop(model, dataloaders, optimizer, criterion, num_epochs, model_path, max_stale=10, negative_only=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    best_auc = -1
    stale_counter = 0
    for epoch in range(num_epochs):
        logging.info(f'## EPOCH {epoch + 1}')
        if stale_counter >= max_stale:
            logging.info(f"Early stopping! Because no improvement for the last {max_stale} epochs...")
            break
        for phase in ['train', 'val']:
            loader = dataloaders[phase]
            start = time.time()

            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
                y_true = []
                y_predicted_probs = []

            with torch.set_grad_enabled(phase == 'train'):
                running_loss = 0.0
                running_correct = 0.0

                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)

                    # restart grads
                    optimizer.zero_grad()

                    # forward
                    out = model(x)

                    # loss
                    loss = criterion(out, y)

                    if phase == 'train':
                        # backward
                        loss.backward()

                        # optimizer step
                        optimizer.step()

                    running_loss += loss
                    _, y_predicted_batch = out.max(1)
                    running_correct += torch.eq(y_predicted_batch, y).sum()

                    if phase == 'val':
                        y_true.extend(y.cpu().tolist())
                        out_probs = F.softmax(out, dim=1)
                        y_predicted_probs.extend(out_probs[:, 1].cpu().tolist())

                logging.info(f'\t{phase}:')

                # avg loss per batch
                epoch_loss = (running_loss / len(loader)).cpu().item()
                history[f'{phase}_loss'].append(epoch_loss)
                logging.info(f'\t\t- Loss: {epoch_loss:.4f}')

                # accuracy in this epoch
                epoch_acc = 100 * running_correct.to(device='cpu', dtype=torch.double).item() / len(loader.dataset)
                history[f'{phase}_acc'].append(epoch_acc)
                logging.info(f'\t\t- Acc: {epoch_acc:.2f}')

                if phase == 'val':
                    # AUC-ROC
                    epoch_auc = roc_auc_score(y_true, y_predicted_probs) if not negative_only else 0
                    history[f'{phase}_auc'].append(epoch_auc)
                    logging.info(f'\t\t- ROC-AUC: {epoch_auc:.4f}')

                    if negative_only:
                        epoch_auc = epoch_acc

                    if epoch_auc > best_auc:
                        best_auc = epoch_auc
                        torch.save(model.state_dict(), f'{model_path}/best_model.pt')
                        stale_counter = 0
                    else:
                        stale_counter += 1

                    logging.info(f'\t\t- best ROC-AUC: {best_auc:.4f}')

                    # save last epoch model
                    torch.save(model.state_dict(), f'{model_path}/last_model.pt')

                    if (epoch + 1) % 2 == 0:
                        save_history(history, model_path)
                        plot_train_curves(history, model_path)

                logging.info(f'\t\t- time: {time.time() - start:.2f} s')


    return history


def save_history(history, model_path):
    df = pd.DataFrame(history)
    df.to_csv(f'{model_path}/history.csv', index=False)


def plot_train_curves(curves_dict, model_path):
    fig, ax = plt.subplots(3, 2)
    curves_names = list(curves_dict.keys())
    for r in range(3):
        for c in range(2):
            i = 2 * r + c
            if i > 4:
                break
            curve_name = curves_names[i]
            curve = curves_dict[curve_name]
            ax[r, c].plot(curve)
            ax[r, c].set_title(curve_name)
            ax[r, c].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'{model_path}/curves')
    plt.close()


def train(model_name, num_epochs, model_path, local, test_run,
          resume_training=None, negative_only=False, max_stale=10):
    lr = 0.001
    batch_size = 16
    logging.info(f'Parameters:\n\t- num_epochs: {num_epochs}\n\t- batch_size: {batch_size}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    if not resume_training:
        model, input_size = initialize_model(model_name)
        model = model.to(device)
    else:
        # todo
        raise NotImplementedError()
        model_path = get_model_path(model_name, local, continue_training)
        mode_file = os.path.join(model_path, '')
        model = None

    # data
    dataloaders = get_dataloaders(input_size, batch_size, local, test_run, negative_only)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss
    criterion = nn.CrossEntropyLoss()

    logging.info(f'Optimizer:\n{optimizer}')
    logging.info(f'Loss function: {criterion}')
    logging.info(f'Model parameters:\n{model}')

    # train loop
    history = train_loop(model, dataloaders, optimizer, criterion, num_epochs, model_path, negative_only=negative_only, max_stale=max_stale)
    save_history(history, model_path)
    plot_train_curves(history, model_path)


def evaluate(local):
    logging.info("Starting evaluation...")


if __name__ == "__main__":
    args = parse_args()
    run_id = get_run_id(args['local'])
    log_name = f"{args['model_name']}_{run_id}.log"
    log_path = f'/scratch/dam740/DLM/logs/{log_name}' if not args['local'] else f'logs/{log_name}'
    model_path = get_model_path(args['model_name'], args['local'], run_id)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s : %(levelname)s : %(message)s ',
                        filemode='w')
    logging.info(json.dumps(args, indent=2))
    logging.info(f"Model path: {model_path}")

    f = train if args['kind'] == 'train' else evaluate
    args['model_path'] = model_path
    del args['kind']
    f(**args)

