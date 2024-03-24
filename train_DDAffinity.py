import os
import shutil
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from torchsummary import summary
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.utils.misc import BlackHole, load_config, seed_all, get_logger, get_new_dir, current_milli_time
from rde.models.protein_mpnn_network_2 import ProteinMPNN_NET
from rde.utils.skempi_mpnn import SkempiDatasetManager
from rde.utils.transforms import get_transform
from rde.utils.train_mpnn import *
from rde.utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--num_cvfolds', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='./logs_skempi')
    parser.add_argument('--early_stoppingdir', type=str, default='./early_stopping')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_dir = None
    else:
        if args.resume:
            log_dir = get_new_dir(args.logdir, prefix='[%d-fold]' % (args.num_cvfolds,), tag=args.tag)
        else:
            log_dir = get_new_dir(args.logdir, prefix='[%d-fold-%d]' % (args.num_cvfolds,config.model.k_neighbors), tag=args.tag)
            early_stoppingdir = get_new_dir(args.early_stoppingdir, prefix='[%d-fold-%d]' % (args.num_cvfolds,config.model.k_neighbors), tag=args.tag)

        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    dataset_mgr = SkempiDatasetManager(
        config,
        num_cvfolds=args.num_cvfolds,
        num_workers=args.num_workers,
        logger=logger,
    )

    # Model, Optimizer & Scheduler
    logger.info('Building model...')
    cv_mgr = CrossValidation(
        model_factory=ProteinMPNN_NET,
        config=config,
        early_stoppingdir=early_stoppingdir,
        num_cvfolds=args.num_cvfolds
    ).to(args.device)
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        cv_mgr.load_state_dict(ckpt['model'], )

    def train(it):
        fold = it % args.num_cvfolds
        model, optimizer, early_stopping = cv_mgr.get(fold)
        if early_stopping.early_stop:
            return

        time_start = current_milli_time()
        model.train()
        # Prepare data
        batch = recursive_to(next(dataset_mgr.get_train_iterator(fold)), args.device)

        # Forward pass
        loss_dict, _ = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        time_backward_end = current_milli_time()

        # Logging
        scalar_dict = {}
        scalar_dict.update({
            'fold': fold,
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger, writer=writer)

        if it >= config.train.early_stop_iters and \
                it % config.train.val_freq == 0 and \
                early_stopping.early_stop == False:
            early_stopping(loss.item(), model, fold)
    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        results = []
        with torch.no_grad():
            for fold in range(args.num_cvfolds):
                results_fold = []
                model, optimizer,_ = cv_mgr.get(fold)

                for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True)):
                    # Prepare data
                    batch = recursive_to(batch, args.device)

                    # Forward pass
                    loss_dict, output_dict = model(batch)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                    for complex, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['complex'], batch["wt"]['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred']):
                        results.append({
                            'complex': complex,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG': ddg_true.item(),
                            'ddG_pred': ddg_pred.item()
                        })
                        results_fold.append({
                            'complex': complex,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG': ddg_true.item(),
                            'ddG_pred': ddg_pred.item()
                        })

                results_fold = pd.DataFrame(results_fold)
                pearson_fold = results_fold[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]


        results = pd.DataFrame(results)
        if ckpt_dir is not None:
            results.to_csv(os.path.join(ckpt_dir, f'results_{it}.csv'), index=False)
        pearson_all = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
        spearman_all = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]

        logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')
        writer.add_scalar('val/all_pearson', pearson_all, it)
        writer.add_scalar('val/all_spearman', spearman_all, it)

        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(it, 'val', logger=logger, writer=writer)

        return avg_loss

    try:
        training_flag = True
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                training_flag = validate(it)
                if training_flag == False:
                    break

        if True:
            results = []
            # Saving 10-fold checkpoint: DDAffinity
            logger.info(f'Saving checkpoint: DDAffinity.pt')
            cv_mgr.save_state_dict(args,config)
            # Loading checkpoint: DDAffinity
            ckpt_path = os.path.join(early_stoppingdir, 'DDAffinity.pt')
            ckpt = torch.load(ckpt_path, map_location=args.device)
            cv_mgr.load_state_dict(ckpt['model'], )

            for fold in range(args.num_cvfolds):
                logger.info(f'Resuming from checkpoint: Fold_{fold}_best_network.pt')
                model, _, _ = cv_mgr.get(fold)
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True)):
                        # Prepare data
                        batch = recursive_to(batch, args.device)

                        # Forward pass
                        loss_dict, output_dict = model(batch)
                        for complex, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['complex'], batch["wt"]['mutstr'],output_dict['ddG_true'],output_dict['ddG_pred']):
                            results.append({
                                'complex': complex,
                                'mutstr': mutstr,
                                'num_muts': len(mutstr.split(',')),
                                'ddG': ddg_true.item(),
                                'ddG_pred': ddg_pred.item()
                            })

            results = pd.DataFrame(results)
            if ckpt_dir is not None:
                results.to_csv(os.path.join(ckpt_dir, f'results_all.csv'), index=False)
            pearson_all = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
            spearman_all = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
            logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')


    except KeyboardInterrupt:
        logger.info('Terminating...')
