import os
import glob
import logging
import time
import json
from parse_args import parse_args
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models
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
import models

NUM_CLASSES = 2


def initialize_resnet(model_name, pretrained):
    if model_name == 'resnet18':
        model_fn = torchvision.models.resnet18
    elif model_name == 'resnet50':
        model_fn = torchvision.models.resnet50
    elif model_name == 'resnet101':
        model_fn = torchvision.models.resnet101
    elif model_name == 'resnet152':
        model_fn = torchvision.models.resnet152

    model = model_fn(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)
    input_size = 224

    return model, input_size


def initialize_densenet(model_name, pretrained):
    if model_name == 'densenet121':
        model_fn = torchvision.models.densenet121
    elif model_name == 'densenet161':
        model_fn = torchvision.models.densenet161
    elif model_name == 'densenet201':
        model_fn = torchvision.models.densenet201

    model = model_fn(pretrained=pretrained)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, NUM_CLASSES)
    input_size = 224

    return model, input_size


def initialize_model(model_name, pretrained=False):
    logging.info(f'Initializing {model_name} model (pretrained? {pretrained}) ...')
    if 'resnet' in model_name:
        model, input_size = initialize_resnet(model_name, pretrained)
        batch_size = 16
    elif 'densenet' in model_name:
        model, input_size = initialize_densenet(model_name, pretrained)
        batch_size = 16
    elif 'basic' in model_name:
        model, input_size = models.BasicCNN(), 96
        batch_size = 50
    return model, input_size, batch_size


def get_dataloaders(input_size, batch_size, kinds=['train', 'val'], local=False, test_run=None, negative_only=False):
    images_path = '/scratch/dam740/DLM/data/images/train'
    test_images_path = '/scratch/dam740/DLM/data/images/test'
    train_labels_path = '/scratch/dam740/DLM/data/train_labels.csv'
    val_labels_path = '/scratch/dam740/DLM/data/val_labels.csv'
    test_labels_path = '/scratch/dam740/DLM/data/test_images.csv'

    if test_run:
        train_labels_path = f"{train_labels_path[:train_labels_path.find('.csv')]}_small.csv"
        val_labels_path = f"{val_labels_path[:val_labels_path.find('.csv')]}_small.csv"
        test_labels_path = f"{test_labels_path[:test_labels_path.find('.csv')]}_small.csv"

    if local:
        images_path = 'data/local_test'
        test_images_path = images_path
        train_labels_path = 'data/small_labels.csv'
        val_labels_path = 'data/small_labels.csv'
        test_labels_path = 'data/small_labels.csv'

    dataloaders = {}
    metadata_paths = {}
    if 'train' in kinds:
        metadata_paths['train'] = train_labels_path
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

        # train data loader
        train_dataset = PcamDataset(images_path, train_labels_path, train_transformations, negative_only)
        dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if 'val' in kinds:
        metadata_paths['val'] = val_labels_path
        # val data loader
        val_dataset = PcamDataset(images_path, val_labels_path, val_transformations, negative_only)
        dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if 'test' in kinds:
        metadata_paths['test'] = test_labels_path
        test_dataset = PcamDataset(test_images_path, test_labels_path, val_transformations)
        dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataloaders, metadata_paths


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


def load_model(model_path, model_name, run_id, device, local):
    # todo: allow to load gpu model into cpu
    model, input_size, batch_size = initialize_model(model_name, pretrained=False)
    weights_path = os.path.join(model_path, 'best_model.pt')
    model.load_state_dict(torch.load(weights_path))
    return model.to(device), input_size, batch_size


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


def eval_loop(model, loader, predictions_only=False, criterion=None, device='cpu'):
    start = time.time()
    model.eval()
    with torch.no_grad():
        y_true = []
        y_predicted_probs = []
        running_loss = 0.0
        running_correct = 0.0
        for inpt, label in loader:
            inpt = inpt.to(device)
            if not predictions_only:
                label = label.to(device)

            # forward
            out = model(inpt)
            out_probs = torch.exp(F.log_softmax(out, dim=1))[:, 1]
            if not predictions_only:
                loss = criterion(out, label)
                running_loss += loss
                _, y_predicted_batch = out.max(1)
                running_correct += torch.eq(y_predicted_batch, label).sum()
                y_true.extend(label.cpu().tolist())

            y_predicted_probs.extend(out_probs.cpu().tolist())

        if not predictions_only:
            # avg loss per batch
            epoch_loss = (running_loss / len(loader)).cpu().item()
            logging.info(f'\t- Loss: {epoch_loss:.4f}')

            # accuracy in this epoch
            epoch_acc = 100 * running_correct.to(device='cpu', dtype=torch.double).item() / len(loader.dataset)
            logging.info(f'\t- Acc: {epoch_acc:.2f}')


            epoch_auc = roc_auc_score(y_true, y_predicted_probs)
            logging.info(f'\t- ROC-AUC: {epoch_auc:.4f}')

        logging.info(f'\t- time: {time.time() - start:.2f} s')

    return y_predicted_probs


def save_submission(predictions, model_path, metadata_path):
    out_path = os.path.join(model_path, 'submission.csv')
    df = pd.read_csv(metadata_path)
    df['label'] = predictions
    df.to_csv(out_path, index=False)

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
          resume_training=None, negative_only=False, max_stale=10,
          pretrained=True):
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    if not resume_training:
        model, input_size, batch_size = initialize_model(model_name, pretrained=pretrained)
        model = model.to(device)
    else:
        model, input_size, batch_size = load_model(model_path, model_name, run_id, device, local)

    logging.info(f'Parameters:\n\t- num_epochs: {num_epochs}\n\t- batch_size: {batch_size}\n\t- max_stale: {max_stale}')
    # data
    dataloaders, _ = get_dataloaders(input_size, batch_size, local=local,
                                  test_run=test_run, negative_only=negative_only)

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


def evaluate(model_path, model_name, run_id, evaluation_kind, local, test_run):
    logging.info("Starting evaluation...")
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, input_size, batch_size = load_model(model_path, model_name, run_id, device, local)
    dataloaders, metadata_paths = get_dataloaders(input_size, batch_size, [evaluation_kind], local, test_run)
    predictions_only = True if evaluation_kind == 'test' else False
    predicted_probs = eval_loop(model, dataloaders[evaluation_kind],
                                predictions_only=predictions_only,
                                criterion=criterion, device=device)
    if evaluation_kind == 'test':
        save_submission(predicted_probs, model_path, metadata_paths['test'])



if __name__ == "__main__":
    args = parse_args()
    if args['kind'] == 'eval':
        run_id = args['run_id']
        model_path = get_model_path(args['model_name'], args['local'], run_id)
        log_path = os.path.join(model_path, 'eval.log')
    else:
        run_id = get_run_id(args['local'])
        model_path = get_model_path(args['model_name'], args['local'], run_id)
        log_name = f"{args['model_name']}_{run_id}.log"
        log_path = f'/scratch/dam740/DLM/logs/{log_name}' if not args['local'] else f'logs/{log_name}'
    args['model_path'] = model_path
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s : %(levelname)s : %(message)s ',
                        filemode='w')
    logging.info(json.dumps(args, indent=2))
    logging.info(f"Model path: {model_path}")

    f = train if args['kind'] == 'train' else evaluate
    del args['kind']
    f(**args)

