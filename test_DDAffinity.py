import os
import copy
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.data import DataLoader, Dataset
from Bio import BiopythonDeprecationWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonDeprecationWarning)
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import Selection
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm
import random
import pickle
import lmdb
from collections import defaultdict
from easydict import EasyDict
from rde.utils.transforms._base import _get_CB_positions

from rde.utils.misc import load_config, seed_all
from rde.utils.data_skempi_mpnn import PaddingCollate
from rde.utils.train_mpnn import *

from rde.utils.transforms import Compose, SelectAtom,AddAtomNoise, SelectedRegionFixedSizePatch
from rde.utils.protein.parsers import parse_biopython_structure
from rde.models.protein_mpnn_network_2 import ProteinMPNN_NET
from rde.utils.skempi_mpnn import  SkempiDatasetManager, eval_skempi_three_modes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--output_results', type=str, default='skempi2_results.csv')
    parser.add_argument('--output_metrics', type=str, default='skempi2_metrics.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--early_stoppingdir', type=str, default='./early_stopping')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_cvfolds', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    config, _ = load_config(args.config)

    # Model
    ckpt = torch.load(config.checkpoint, map_location=args.device)
    config_model = ckpt['config']
    num_cvfolds = len(ckpt['model']['models'])
    cv_mgr = CrossValidation(
        model_factory=ProteinMPNN_NET,
        config=config_model,
        early_stoppingdir=config.early_stoppingdir,
        num_cvfolds=num_cvfolds
    ).to(args.device)
    cv_mgr.load_state_dict(ckpt['model'], )
    print(config_model)

    # Data
    dataset_mgr = SkempiDatasetManager(
        config_model,
        num_cvfolds=num_cvfolds,
        num_workers=args.num_workers,
    )

    results = []
    for fold in range(num_cvfolds):
        model, _, _ = cv_mgr.get(fold)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                _, output_dict = model(batch)
                for complex, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['complex'], batch["wt"]['mutstr'],
                                                               output_dict['ddG_true'], output_dict['ddG_pred']):
                    results.append({
                        'complex': complex,
                        'mutstr': mutstr,
                        'num_muts': len(mutstr.split(',')),
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })


    results = pd.DataFrame(results)

    results['datasets'] = 'SKEMPI2'
    results.to_csv(args.output_results, index=False)

    results = pd.read_csv(args.output_results)
    # PDB:1E96.pdb and 1E50.pdb
    results.replace("1.00E+96", "1E96", inplace = True)
    results.replace("1.00E+50", "1E50", inplace = True)

    # 显示所有列,保留3位小数
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = '{:.3f}'.format

    results['datasets'] = 'SKEMPI2'
    df_metrics = eval_skempi_three_modes(results)
    df_metrics.to_csv(args.output_metrics, index=False)
    print(df_metrics)

