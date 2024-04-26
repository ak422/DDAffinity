import os
import copy
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.data import DataLoader, Dataset
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
# from rde.models.rde_ddg import DDG_RDE_Network
from rde.models.protein_mpnn_network_2 import ProteinMPNN_NET
from rde.utils.skempi_mpnn import  eval_skempi_three_modes


class CaseDataset(Dataset):
    MAP_SIZE = 500 * (1024 * 1024 * 1024)  # 500GB
    def __init__(self, pdb_wt_path, pdb_mt_path, cache_dir, mutations):
        super().__init__()
        self.pdb_wt_path = pdb_wt_path
        self.pdb_mt_path = pdb_mt_path
        self.cache_dir = cache_dir

        self.data = []
        self.db_conn = None
        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self._load_entries(reset=False)

        self.mutations = self._parse_mutations(mutations)

        self.transform = Compose([
            SelectAtom('backbone+CB'),
            SelectedRegionFixedSizePatch('mut_flag', 256)
        ])
    def save_mutations(self,mutations):
        # data1为list类型，参数index为索引，column为列名
        data2 = pd.DataFrame(data=mutations, index=None)
        # PATH为导出文件的路径和文件名
        path = os.path.join(os.path.dirname(self.pdb_path),"7FAE_RBD_Fv_mutations.csv")
        data2.to_csv(path)

    def clone_data(self):
        return copy.deepcopy(self.data)

    def _load_entries(self, reset):
        with open(self.entries_cache, 'rb') as f:
            self.entries = pickle.load(f)

    def _load_data(self, pdb_wt_path, pdb_mt_path):
        self.data.append(self._load_structure(pdb_wt_path))
        self.data.append(self._load_structure(pdb_mt_path))

    def _load_structure(self, pdb_path):
        if pdb_path.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        elif pdb_path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError('Unknown file type.')

        model = parser.get_structure(None, pdb_path)[0]
        chains = Selection.unfold_entities(model, 'C')
        random.shuffle(chains)  # shuffle chains，增加数据多样性

        data, seq_map = parse_biopython_structure(model, chains)
        return data, seq_map

    def _parse_mutations(self, mutations):
        parsed = []
        for m in mutations:
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])

            if mt == '*':
                for mt_idx in range(20):
                    mt = index_to_one(mt_idx)
                    if mt == wt: continue
                    parsed.append({
                        'chain': ch,
                        'seq': seq,
                        'wt': wt,
                        'mt': mt,
                    })
            else:
                parsed.append({
                    'chain': ch,
                    'seq': seq,
                    'wt': wt,
                    'mt': mt,
                })
        return parsed

    @property
    def lmdb_path(self):
        return os.path.join(self.cache_dir, 'structures.lmdb')

    @property
    def keys_path(self):
        return os.path.join(self.cache_dir, 'keys.pkl')
    def _connect_db(self):
        assert self.db_conn is None
        self.db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with open(self.keys_path, 'rb') as f:
            self.db_keys = pickle.load(f)
    def _close_db(self):
        self.db_conn.close()
        self.db_conn = None
        self.db_keys = None
    def _get_from_db(self, pdbcode):
        if self.db_conn is None:
            self._connect_db()
        data = pickle.loads(self.db_conn.begin().get(pdbcode.encode()))   # Made a copy
        return data
    def _compute_degree_centrality(
            self,
            data,
            atom_ty="CB",
            dist_thresh=10,
    ):
        pos_beta_all = _get_CB_positions(data['pos_heavyatom'], data['mask_heavyatom'])
        pairwise_dists = torch.cdist(pos_beta_all, pos_beta_all)
        return torch.sum(pairwise_dists < dist_thresh, dim=-1) - 1
    def __len__(self):
        # return len(self.mutations)
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]  # 按蛋白质复合物结构读取
        pdbcode = entry['pdbcode']

        data_wt, seq_map_wt = self._get_from_db("wt_" + pdbcode)  # Made a copy
        data_mt, seq_map_mt = self._get_from_db("mt_" + pdbcode)  # Made a copy

        data_dict_wt = defaultdict(list)
        data_dict_mt = defaultdict(list)
        chains_list = list(data_wt.keys())
        random.shuffle(chains_list)  # 将链随机
        for i in chains_list:
            if isinstance(data_wt[i], EasyDict):
                for k, v in data_wt[i].items():
                    data_dict_wt[k].append(v)
            if isinstance(data_mt[i], EasyDict):
                for k, v in data_mt[i].items():
                    data_dict_mt[k].append(v)

        for k, v in data_dict_wt.items():
            data_dict_wt[k] = torch.cat(data_dict_wt[k], dim=0)
        for k, v in data_dict_mt.items():
            data_dict_mt[k] = torch.cat(data_dict_mt[k], dim=0)

        # centrality
        # pos_heavyatom: ['N', 'CA', 'C', 'O', 'CB']
        data_dict_wt['centrality'] = self._compute_degree_centrality(data_dict_wt)  # CB原子
        data_dict_mt['centrality'] = self._compute_degree_centrality(data_dict_mt)  # CB原子

        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'ddG'}
        for k in keys:
            data_dict_wt[k] = data_dict_mt[k] = entry[k]

        assert len(entry['mutations']) == torch.sum(data_dict_wt['aa'] != data_dict_mt[
            'aa']), f"ID={data_dict_wt['id']},{len(entry['mutations'])},{torch.sum(data_dict_wt['aa'] != data_dict_mt['aa'])}"
        data_dict_wt['mut_flag'] = data_dict_mt['mut_flag'] = (data_dict_wt['aa'] != data_dict_mt['aa'])

        if self.transform is not None:
            data_dict_wt, idx_mask = self.transform(data_dict_wt)
            data_dict_mt, _ = self.transform(data_dict_mt, idx_mask)

        return {"wt": data_dict_wt,
                "mt": data_dict_mt
                }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--output_results', type=str, default='case_results.csv')
    parser.add_argument('--output_metrics', type=str, default='case_metrics.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    config, _ = load_config(args.config)

    # Model
    ckpt = torch.load(config.checkpoint, map_location=args.device)
    config_model = ckpt['config']
    num_cvfolds = len(ckpt['model']['models'])
    cv_mgr = CrossValidation(model_factory=ProteinMPNN_NET, config=config_model, early_stoppingdir=config.early_stoppingdir, num_cvfolds=num_cvfolds).to(args.device)
    cv_mgr.load_state_dict(ckpt['model'], )
    print(config_model)

    # Data
    dataset = CaseDataset(
        pdb_wt_path = config.pdb_wt_dir,
        pdb_mt_path = config.pdb_mt_dir,
        cache_dir = config.cache_dir,
        mutations = config.mutations,
    )
    loader = DataLoader(
        dataset, 
        batch_size=config_model.train.batch_size,
        shuffle=False, 
        collate_fn=PaddingCollate(config_model.model.patch_size),
    )

    results = []
    for batch in tqdm(loader):
        batch = recursive_to(batch, args.device)
        for fold in range(cv_mgr.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()

            with torch.no_grad():
                _, out_dict = model(batch)
                ddg_pred = out_dict['ddG_pred']

            for pdbcode, mutstr, ddG_pred,ddG in zip(batch['wt']['pdbcode'], batch['wt']['mutstr'], ddg_pred.squeeze(-1).cpu().tolist(),batch["wt"]['ddG'].squeeze(-1).cpu().tolist()):
                results.append({
                    'pdbcode': pdbcode,
                    'mutstr': mutstr,
                    'num_muts': len(mutstr.split(',')),
                    'ddG': ddG,
                    'ddG_pred': ddG_pred,
                })
    results = pd.DataFrame(results)
    results.to_csv(args.output_results, index=False)

    results = pd.read_csv(args.output_results)
    if 'M595' in config.cache_dir:
        # # : M595
        results = results.groupby('pdbcode').agg(ddG_pred_max= ("ddG_pred", "max"),
                                                ddG=("ddG", "mean"),
                                                num_muts=("num_muts", "mean")).reset_index()
        results['ddG_pred'] = results['ddG_pred_max']
        results['datasets'] = 'case_study'
        df_metrics = eval_skempi_three_modes(results)
        df_metrics.to_csv(args.output_metrics)
        print(df_metrics)
    elif 'S285' in config.cache_dir:
        # # : S285
        results = results.groupby('mutstr').agg(ddG_pred_max= ("ddG_pred", "max"),
                                                ddG=("ddG", "mean"),
                                                num_muts=("num_muts", "mean")).reset_index()
        results['ddG_pred'] = - results['ddG_pred_max']
        results['datasets'] = 'case_study'
        df_metrics = eval_skempi_three_modes(results)
        df_metrics.to_csv(args.output_metrics)
        print(df_metrics)
    elif 'S494' in config.cache_dir:
        # # : S494
        results = results.groupby('mutstr').agg(ddG_pred_max=("ddG_pred", "max"),
                                                ddG=("ddG", "mean"),
                                                num_muts=("num_muts", "mean")).reset_index()
        results['rank'] = (results['ddG_pred_max']).rank() / len(results)
        results['datasets'] = 'case_study'
        if 'interest' in config and config.interest:
            print(results[results['mutstr'].isin(config.interest)])
