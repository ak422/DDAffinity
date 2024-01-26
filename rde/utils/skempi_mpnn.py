import functools
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import torch
import os
import pickle
import random

from rde.utils.misc import inf_iterator, BlackHole
from rde.utils.data_skempi_mpnn import PaddingCollate
from rde.utils.transforms import get_transform
from rde.datasets import SkempiDataset
from rde.datasets import SkempiDataset_lmdb

from torch.utils.data import Sampler,BatchSampler
class batch_sampler(Sampler):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        chains = self.get_chains()
        cluster = [(i, chains[e['pdbcode']]) for i,e in enumerate(self.dataset.entries)]
        cluster.sort(key=lambda x: x[1])

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix, size in cluster:
            if size * (len(batch) + 1) <= config.data.residue_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def get_chains(self):
        chains_path = os.path.join(self.config.data.cache_dir, 'chains.pkl')
        with open(chains_path, 'rb') as f:
            chains = pickle.load(f)
        return chains

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        random.shuffle(self.clusters)
        batch = []
        for b_idx in self.clusters:
            random.shuffle(b_idx)
            batch = b_idx
            yield batch
            batch = []

class SkempiDatasetManager(object):

    def __init__(self, config, num_cvfolds, num_workers=4, logger=BlackHole()):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.train_iterators = []
        self.val_loaders = []
        self.chains = []
        self.logger = logger
        self.num_workers = num_workers
        for fold in range(num_cvfolds):
            train_iterator, val_loader = self.init_loaders(fold)
            self.train_iterators.append(train_iterator)
            self.val_loaders.append(val_loader)

    def init_loaders(self, fold):
        config = self.config
        dataset_ = functools.partial(
            SkempiDataset_lmdb,
            csv_path = config.data.csv_path,
            pdb_wt_dir = config.data.pdb_wt_dir,
            pdb_mt_dir=config.data.pdb_mt_dir,
            cache_dir = config.data.cache_dir,
            num_cvfolds = self.num_cvfolds,
            cvfold_index = fold,
        )

        train_dataset = dataset_(split='train',transform = get_transform(config.data.train.transform))
        val_dataset = dataset_(split='val',transform = get_transform(config.data.val.transform))
        
        train_cplx = set([e['complex'] for e in train_dataset.entries])
        val_cplx = set([e['complex'] for e in val_dataset.entries])
        leakage = train_cplx.intersection(val_cplx)
        assert len(leakage) == 0, f'data leakage {leakage}'

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.train.batch_size,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            shuffle=True,
            num_workers=self.num_workers
        )
        train_iterator = inf_iterator(train_loader)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.train.batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            num_workers=self.num_workers
        )
        self.logger.info('Fold %d: Train %d, Val %d' % (fold, len(train_dataset), len(val_dataset)))
        return train_iterator, val_loader

    def get_train_iterator(self, fold):
        return self.train_iterators[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]

def overall_correlations(df):
    pearson = df[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1]
    spearman = df[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1]
    return {
        'overall_pearson': pearson, 
        'overall_spearman': spearman,
    }

def overall_auroc(df):
    score = roc_auc_score(
        (df['ddG'] > 0).to_numpy(),
        df['ddG_pred'].to_numpy()
    )
    return {
        'auroc': score,
    }


def overall_rmse_mae(df):
    true = df['ddG'].to_numpy()
    pred = df['ddG_pred'].to_numpy()[:, None]
    reg = LinearRegression().fit(pred, true)
    pred_corrected = reg.predict(pred)
    rmse = np.sqrt( ((true - pred_corrected) ** 2).mean() )
    mae = np.abs(true - pred_corrected).mean()
    return {
        'rmse': rmse,
        'mae': mae,
    }


def analyze_all_results(df):
    datasets = df['datasets'].unique()
    funcs = {
        'SKEMPI2': [overall_correlations,
                    overall_rmse_mae,
                    overall_auroc],
        'case_study': [overall_correlations,
                 overall_rmse_mae,
                 overall_auroc]
    }
    analysis = []
    for dataset in tqdm(datasets):
        assert dataset in ['SKEMPI2', 'case_study']
        df_this = df[df['datasets'] == dataset]
        result = {
            'dataset': dataset,
        }
        for f in funcs[dataset]:
            result.update(f(df_this))
        analysis.append(result)
    analysis = pd.DataFrame(analysis)
    return analysis

def eval_skempi(df_items, mode, ddg_cutoff=None):
    assert mode in ('all', 'single', 'multiple')
    if mode == 'single':
        df_items = df_items.query('num_muts == 1')
    elif mode == 'multiple':
        df_items = df_items.query('num_muts > 1')

    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")

    df_metrics = analyze_all_results(df_items)
    df_metrics['mode'] = mode
    return df_metrics


def eval_skempi_three_modes(results, ddg_cutoff=None):
    df_all = eval_skempi(results, mode='all', ddg_cutoff=ddg_cutoff)
    df_single = eval_skempi(results, mode='single', ddg_cutoff=ddg_cutoff)
    df_multiple = eval_skempi(results, mode='multiple', ddg_cutoff=ddg_cutoff)
    df_metrics = pd.concat([df_all, df_single, df_multiple], axis=0)
    df_metrics.reset_index(inplace=True, drop=True)
    return df_metrics
