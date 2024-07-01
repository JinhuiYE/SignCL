# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

# *user-defined
from models import SLRCLIP, Text_Decoder
import utils as utils
from datasets import S2T_Dataset

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import wandb
import copy
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import SoftTargetCrossEntropy
from timm.optim import AdamW

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image

from hpman.m import _
import hpargparse

# global definition
from definition import *

import gzip
import pickle
import torch

import signcl as signcl
cl_criterion = signcl.SignCL()

def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free_csl.yaml')

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)

    # * wandb params
    parser.add_argument("--log_all", action="store_true",
                        help="flag to log in all processes, otherwise only in rank0",
                        )
    parser.add_argument("--entity", type=str,
                        help="wandb entity",
                        )
    parser.add_argument("--project", type=str, default='VLP',
                        help="wandb project",
                        )

    # * Noise params
    parser.add_argument('--training-refurbish', default=True, type=bool)
    parser.add_argument('--noise-rate', default=0.15, type=float)
    parser.add_argument('--noise-type', default='omit_last', type=str, choices=['omit', 'omit_last'])
    parser.add_argument('--random-shuffle', default=False, type=bool)

    parser.add_argument('--loss-lambda', type=float, default=1.0, metavar='RATE',
                        help='lambda param')

    return parser


def main(args, config):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['transformer'])

    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer=tokenizer, config=config, args=args,
                             phase='train', training_refurbish=True)
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn,
                                  sampler=train_sampler,
                                  pin_memory=args.pin_mem,
                                  drop_last=True)

    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer=tokenizer, config=config, args=args,
                           phase='val', training_refurbish=True)
    print(dev_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data, shuffle=False)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                sampler=dev_sampler,
                                pin_memory=args.pin_mem)

    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer=tokenizer, config=config, args=args,
                            phase='test', training_refurbish=True)
    print(test_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler,
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    model = SLRCLIP(config=config)
    model.to(device)
    print(model)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    text_decoder = Text_Decoder(config).to(device)

    if args.distributed:
        text_decoder = torch.nn.parallel.DistributedDataParallel(text_decoder, device_ids=[args.gpu],
                                                                 find_unused_parameters=True)
    optimizer_td = AdamW(text_decoder.module.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98))

    lr_scheduler_td = scheduler.CosineAnnealingLR(
        optimizer=optimizer_td,
        eta_min=1e-8,
        T_max=args.epochs,
    )
    TD_train_dict = dict(
        optimizer=optimizer_td,
        lr_scheduler=lr_scheduler_td,
        text_decoder=text_decoder
    )

    criterion = utils.KLLoss()
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, args.start_epoch,
                             UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, criterion, config, args.start_epoch,
                              UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config,
                                      PAD_IDX, loss_scaler, TD_train_dict)
        lr_scheduler.step(epoch)
        TD_train_dict['lr_scheduler'].step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
                              SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)

        if min_loss > test_stats["loss"]:
            min_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                        'epoch': epoch,
                        # 'args': args,
                    }, checkpoint_path)

        print(f"* DEV loss {test_stats['loss']:.3f} Min DEV loss {min_loss}")
        if args.run:
            args.run.log({'epoch': epoch + 1, 'training/train_loss': train_stats['loss'],
                          'training/masked_lm_loss': train_stats['masked_lm_loss'], 'dev/dev_loss': test_stats['loss'],
                          'dev/min_loss': min_loss})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        torch.distributed.barrier()
        checkpoint = torch.load(args.output_dir + '/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
                             SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
                              SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, PAD_IDX, loss_scaler, TD_train_dict, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    for step, (src_input, tgt_input, masked_tgt_input) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text, ground_truth, frames_feature = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            margin = max(10, int((frames_feature.shape[1] // tgt_input['input_ids'].shape[1] + 1) * 2.3)) * 2
            num_negative = 30
            margin = min(margin, int((frames_feature.shape[1] - num_negative) / 2)) #ensure num_frames margin for negative sampling
            cl_loss = cl_criterion(frames_feature, margin=margin)

            ml_loss = (loss_imgs + loss_texts) / 2.
            total_loss = ml_loss + 0.01 * cl_loss
        loss_scaler(ml_loss, optimizer)

        # update the text decoder parames
        if step % 5 == 0:
            TD_train_dict['optimizer'].zero_grad()
            with torch.cuda.amp.autocast():
                lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.module.model_txt)
                masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]),
                                          tgt_input['input_ids'].cuda().view(-1)) * args.loss_lambda
            loss_scaler(masked_lm_loss, TD_train_dict['optimizer'])

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(cl_loss=cl_loss.item())
        metric_logger.update(masked_lm_loss=masked_lm_loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(td_lr=TD_train_dict['optimizer'].param_groups[0]["lr"])

        if (step + 1) % 10 == 0 and utils.is_main_process():
            visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
            utils.visualization([visual_map, ])

    if args.run:
        args.run.log(
            {'epoch': epoch + 1, 'epoch/train_loss': loss_value, 'epoch/masked_lm_loss': masked_lm_loss.item()})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS,
             PAD_IDX, device, TD_train_dict):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    with torch.no_grad():
        for step, (src_input, tgt_input, masked_tgt_input) in enumerate(
                metric_logger.log_every(dev_dataloader, print_freq, header)):

            logits_per_image, logits_per_text, ground_truth, frames_feature = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model_without_ddp.model_txt)
            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1))
            total_loss = (loss_imgs + loss_texts) / 2.

            metric_logger.update(loss=total_loss.item())
            metric_logger.update(masked_lm_loss=masked_lm_loss.item())

            if (step + 1) % 10 == 0 and utils.is_main_process():
                visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
                utils.visualization([visual_map, ])

    if args.run:
        args.run.log({'epoch': epoch + 1, 'epoch/dev_loss': total_loss.item()})

    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def extract_sign_features(args, dev_dataloader, model, output_file, split="dev"):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    image_features_dict = {}

    with torch.no_grad():
        for images, _ in metric_logger.log_every(dev_dataloader, 20):
            image_features = model.get_images_feature(
                src_input=images)  # Assuming model has a method to extract features
            image_features = image_features.cpu()  # Convert to numpy and store
            for index, name in enumerate(images["name_batch"]):
                image_features_dict[name] = image_features[index]

    metric_logger.synchronize_between_processes()
    print("* Extracted features for dataset")
    path = f"./GFSLT-VLP/data/Phonexi-2014T/labels.{split}"
    raw_data = utils.load_dataset_file(path)
    datalist = []
    for name in raw_data.keys():
        item = raw_data[name]
        if name in image_features_dict.keys():
            item["sign"] = image_features_dict[name]
            item["signer"] = "Signer99"

            datalist.append(item)
    # Optionally, save the extracted features to a file
    with gzip.open(output_file, "wb") as f:
        pickle.dump(datalist, f)


def main_extract_features(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Dataset and DataLoader setup
    print("Preparing data...")
    split = "train"
    tokenizer = MBartTokenizer.from_pretrained(config['model']['transformer'])
    dev_data = S2T_Dataset(path=config['data'][f'{split}_label_path'], tokenizer=tokenizer, config=config, args=args,
                           phase='val')
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn_wname, pin_memory=args.pin_mem)

    # Model setup
    print("Loading model...")
    model = SLRCLIP(config=config).to(device)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        # Filter and load FeatureExtracter parameters
        feature_extracter_params = {k: v for k, v in checkpoint['model'].items() if
                                    'model_images.model.' in k}
        model.load_state_dict(feature_extracter_params, strict=False)

    # Extract and save features
    # output_file = os.path.join(args.output_dir, f"{split}_extracted_features")

    output_directory = "./GFSLT-VLP/data/Phonexi-2014T/GFSLT_VLP_SignCL/"
    output_file = f"{output_directory}phoenix14t.pami0.{split}"
    # 创建目录（如果不存在）
    os.makedirs(output_directory, exist_ok=True)

    extract_sign_features(args, dev_dataloader, model, output_file, split=split)
    print(f"Features extracted and saved to {output_file}")


def setup_run(args, config):
    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1]
        else:
            os.environ["WANDB_MODE"] = 'disabled'
            run = False

    return run


if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
    # main_extract_features(args, config)