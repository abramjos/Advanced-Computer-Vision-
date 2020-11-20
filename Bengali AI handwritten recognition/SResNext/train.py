from data_gen import *
import configurations as cfg
from model import PretrainedCNN, BengaliClassifier
from utils.transform import Transform
from utils.metric import macro_recall
from utils.ignite_helper import *

import argparse
from distutils.util import strtobool

import os
import torch

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from numpy.random.mtrand import RandomState
from torch.utils.data.dataloader import DataLoader

# Getting the dataset for training set
train = pd.read_csv(cfg.datadir/'train.csv')
train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

# Creating the folds
indices = [0] if cfg.debug else [0, 1, 2, 3]
train_images = prepare_image(
    cfg.datadir, cfg.featherdir, data_type='train', submission=False, indices=indices)

# Creating the dataset split
n_dataset = len(train_images)
train_data_size = 200 if cfg.debug else int(n_dataset * 0.8)
valid_data_size = 100 if cfg.debug else int(n_dataset - train_data_size)
perm = np.random.RandomState(777).permutation(n_dataset)

#Training and Validation set
train_transform = Transform(
    size=(cfg.image_size, cfg.image_size), threshold=20.,
    sigma=-1., blur_ratio=0.2, noise_ratio=0.2, cutout_ratio=0.2,
    grid_distortion_ratio=0.2, random_brightness_ratio=0.2,
    piece_affine_ratio=0.2, ssr_ratio=0.2)

train_dataset = BengaliAIDataset(
    train_images, train_labels, transform=train_transform,indices=perm[:train_data_size])
    #train_images, train_labels, transform=Transform(size=(cfg.image_size, cfg.image_size)), indices=perm[:train_data_size])

valid_dataset = BengaliAIDataset(
    train_images, train_labels, transform=Transform(affine=False, crop=True, size=(cfg.image_size, cfg.image_size)),
    indices=perm[train_data_size:train_data_size+valid_data_size])

print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))


train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)


device = torch.device(cfg.device)
n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant


predictor = PretrainedCNN(in_channels=1, out_dim=n_total, model_name=cfg.model_name, pretrained=None)
classifier = BengaliClassifier(predictor).to(cfg.device)


optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)


trainer = create_trainer(classifier, optimizer, cfg.device)
def output_transform(output):
    metric, pred_y, y = output
    return pred_y.cpu(), y.cpu()
EpochMetric(
    compute_fn=macro_recall,
    output_transform=output_transform
).attach(trainer, 'recall')


pbar = ProgressBar()
pbar.attach(trainer, metric_names='all')

evaluator = create_evaluator(classifier, cfg.device)
EpochMetric(
    compute_fn=macro_recall,
    output_transform=output_transform
).attach(evaluator, 'recall')

def run_evaluator(engine):
    evaluator.run(valid_loader)

def schedule_lr(engine):
    # metrics = evaluator.state.metrics
    metrics = engine.state.metrics
    avg_mae = metrics['loss']

    # --- update lr ---
    lr = scheduler.optimizer.param_groups[0]['lr']
    scheduler.step(avg_mae)
    log_report.report('lr', lr)

trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)
log_report = LogReport(evaluator, cfg.outdir)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    ModelSnapshotHandler(predictor, filepath=cfg.outdir / 'predictor.pt'))

trainer.run(train_loader, max_epochs=100)

train_history = log_report.get_dataframe()
train_history.to_csv(cfg.outdir / 'log.csv', index=False)


