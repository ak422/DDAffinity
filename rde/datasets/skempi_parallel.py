import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import sys
sys.path.append('../..')
import os
import copy
import random
import pickle
import math
import torch
import lmdb
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import one_to_index
from rde.utils.protein.dihedral_chi import get_dihedral
from rde.utils.misc import  current_milli_time
from easydict import EasyDict
from collections import defaultdict
from typing import Mapping, List, Dict, Tuple, Optional
from joblib import Parallel, delayed, cpu_count
from scipy import stats, special
from rde.utils.transforms._base import _get_CB_positions

from rde.utils.protein.parsers import parse_biopython_structure

PERTURB_NUM = 50

def _get_structure(pdb_path, chains, pdbcode, flag):
    parser = PDBParser(QUIET=True)
    model = parser.get_structure(None, pdb_path)[0]
    data, seq_map = parse_biopython_structure(model, chains)
    _structures = {}
    _structures[flag + "_" +pdbcode] = (data, seq_map)

    return _structures

def best_lmbda(y_true):
    from scipy.stats import boxcox_normmax
    eps = 0.1
    best_lmbda = boxcox_normmax(y_true - np.min(y_true)+ eps)
    return best_lmbda

def load_skempi_entries(csv_path, pdb_wt_dir, pdb_mt_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=';')
    df['dG_wt'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    df['ddG'] = df['dG_mut'] - df['dG_wt']

    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    entries = []
    for i, row in df.iterrows():
        pdbcode, group1, group2 = row['#Pdb'].split('_')
        if pdbcode in block_list:
            continue
        mut_str = row['Mutation(s)_cleaned']
        mut_list = set(mut_str.split(','))
        muts = list(map(_parse_mut, mut_list))
        if muts[0]['chain'] in group1:
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        pdb_wt_path = os.path.join(pdb_wt_dir, '{}_{}.pdb'.format(str(i), pdbcode.upper()))
        pdb_mt_path = os.path.join(pdb_mt_dir, '{}_{}.pdb'.format(str(i), pdbcode.upper()))
        if not os.path.exists(pdb_wt_path) or not os.path.exists(pdb_mt_path):
            continue

        if not np.isfinite(row['ddG']):
            continue

        entry = {
            'id': i,
            'complex': row['#Pdb'],
            'mutstr': mut_str,
            'num_muts': len(muts),
            'pdbcode': str(i)+"_"+pdbcode,
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': muts,
            'ddG': np.float32(row['ddG']),
            # 'ddG_min': np.float32(row['ddG_min']),
            # 'fitted_lambda': np.float32(row['fitted_lambda']),
            'pdb_wt_path': pdb_wt_path,
            'pdb_mt_path': pdb_mt_path,
        }
        entries.append(entry)

    # # transform training data & save lambda value
    # ddG = []
    # for entry_item in entries:
    #     ddG.append(entry_item["ddG"])
    # ddG = np.array(ddG)
    # eps = 0.1
    # lmbda = best_lmbda(ddG)
    # fitted_data = special.boxcox1p(ddG - np.min(ddG) + eps, lmbda)
    # for i, entry_item in enumerate(entries):
    #     entry_item['fitted_data'] = fitted_data[i]

    return entries
def load_category_entries(csv_path, pdb_wt_dir, pdb_mt_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=',')
    # df['dG_wt'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    # df['dG_mut'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    # df['ddG'] = df['dG_mut'] - df['dG_wt']

    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }
    def _parse_reverse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        mut_name = "".join([mt_type, mutchain, mut_name[2:-1], wt_type])
        return {
            'wt': mt_type,
            'mt': wt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    entries = []
    for i, row in df.iterrows():
        pdbcode = row['#Pdb']
        group1 = row['Partner1']
        group2 = row['Partner2']

        if pdbcode.split("_")[1] in block_list:
            continue
        mut_str = row['Mutation(s)_cleaned']
        mut_list = set(mut_str.split(','))

        if row['Label'] == "forward":
            muts = list(map(_parse_mut, mut_list))
        else:
            # 处理S1707的逆突变
            muts = list(map(_parse_reverse_mut, mut_list))
            mut_str = ",".join([mut['name'] for mut in muts])
            row['ddG'] = -row['ddG']

        if muts[0]['chain'] in group1:
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        if row['Label'] == "forward":
            pdb_wt_path = os.path.join(pdb_wt_dir, '{}.pdb'.format(pdbcode.upper()))
            pdb_mt_path = os.path.join(pdb_mt_dir, '{}.pdb'.format(pdbcode.upper()))
        else:    # 处理S1707的逆突变
            pdb_mt_path = os.path.join(pdb_wt_dir, '{}.pdb'.format(pdbcode.upper()))
            pdb_wt_path = os.path.join(pdb_mt_dir, '{}.pdb'.format(pdbcode.upper()))

        if not os.path.exists(pdb_wt_path) or not os.path.exists(pdb_mt_path):
            continue

        if not np.isfinite(row['ddG']):
            continue

        entry = {
            'id': i,
            'complex': pdbcode.split("_")[1],
            'mutstr': mut_str,
            'num_muts': len(muts),
            'pdbcode': pdbcode.upper(),
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': muts,
            'ddG': np.float32(row['ddG']),
            'pdb_wt_path': pdb_wt_path,
            'pdb_mt_path': pdb_mt_path,
        }
        entries.append(entry)

    return entries

def _process_structure(pdb_wt_path, pdb_mt_path, pdbcode) -> Optional[Dict]:
    structures = defaultdict(dict)
    parser = PDBParser(QUIET=True)
    model = parser.get_structure(None, pdb_wt_path)[0]
    chains = Selection.unfold_entities(model, 'C')

    # delete invalid chain
    for i, chain in enumerate(chains):
        if chain.id == " ":
            del chains[i]

    random.shuffle(chains)  # shuffle chains，增加数据多样性
    structures.update(_get_structure(pdb_wt_path, chains, pdbcode, "wt"))
    structures.update(_get_structure(pdb_mt_path, chains, pdbcode, "mt"))
    return structures


class SkempiDataset_lmdb(Dataset):

    MAP_SIZE = 500 * (1024 * 1024 * 1024)  # 500GB
    def __init__(
        self, 
        csv_path, 
        pdb_wt_dir,
        pdb_mt_dir,
        cache_dir,
        cvfold_index=0, 
        num_cvfolds=3, 
        split='train', 
        split_seed=2022,
        num_preprocess_jobs=math.floor(cpu_count() * 0.6),
        transform=None, 
        blocklist=frozenset({'1KBH'}), 
        reset=False,
        # ak422
        is_single=2,  # 0:single,1:multiple,2:overall
    ):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_wt_dir = pdb_wt_dir
        self.pdb_mt_dir = pdb_mt_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        self.num_cvfolds = num_cvfolds
        assert split in ('train', 'val')
        self.split = split
        self.split_seed = split_seed
        self.num_preprocess_jobs = num_preprocess_jobs

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None                 # 按训练集和测试集划分
        self.entries_full = None            # 按数据集划分
        # zqc
        self.is_single = is_single
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.lmdb')
        self.structures = None
        # Structure cache
        self.db_conn = None
        self.db_keys: Optional[List[PdbCodeType]] = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()    # 按数据集划分
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        complex_to_entries = {}
        for e in self.entries_full:
            if e['complex'] not in complex_to_entries:
                complex_to_entries[e['complex']] = []
            complex_to_entries[e['complex']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        random.Random(self.split_seed).shuffle(complex_list)

        split_size = math.ceil(len(complex_list) / self.num_cvfolds)
        complex_splits = [
            complex_list[i*split_size : (i+1)*split_size] 
            for i in range(self.num_cvfolds)
        ]

        val_split = complex_splits.pop(self.cvfold_index)
        train_split = sum(complex_splits, start=[])
        if self.split == 'val':
            complexes_this = val_split
        else:
            complexes_this = train_split

        entries = []
        for cplx in complexes_this:
            #  single or multiple
            if self.is_single == 0:
                for complex_item in complex_to_entries[cplx]:
                    if complex_item['num_muts'] > 1:
                        continue
                    else:
                        entries += [complex_item]
            elif self.is_single == 1:
                for complex_item in complex_to_entries[cplx]:
                    if complex_item['num_muts'] == 1:
                        continue
                    else:
                        entries += [complex_item]
            else:
                entries += complex_to_entries[cplx]

        self.entries = entries
        
    def _preprocess_entries(self):
        entries_full = load_category_entries(self.csv_path, self.pdb_wt_dir, self.pdb_mt_dir, self.blocklist)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries_full, f)      # 按数据集划分
        return entries_full

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures(reset)
        else:
            return None
            # with open(self.structures_cache, 'rb') as f:
            #     self.structures = pickle.load(f)

    @property
    def lmdb_path(self):
        return os.path.join(self.cache_dir, 'structures.lmdb')
    @property
    def keys_path(self):
        return os.path.join(self.cache_dir, 'keys.pkl')
    @property
    def chains_path(self):
        return os.path.join(self.cache_dir, 'chains.pkl')

    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path) and not reset:
            return

        pdbcodes = list(set([(e['pdbcode'],e['pdb_wt_path'],e['pdb_mt_path']) for e in self.entries_full]))  # 这里是按结构process：wt & mt
        pdbcodes.sort()
        tasks = []
        for (pdbcode,pdb_wt_path,pdb_mt_path)  in tqdm(pdbcodes, desc='Structures'):
            if not os.path.exists(pdb_wt_path):
                print(f'[WARNING] PDB not found: {pdb_wt_path}.')
                continue
            if not os.path.exists(pdb_mt_path):
                print(f'[WARNING] PDB not found: {pdb_mt_path}.')
                continue

            tasks.append(
                delayed(_process_structure)(pdb_wt_path, pdb_mt_path, pdbcode)
            )

        # Split data into chunks
        chunk_size = 512
        task_chunks = [
            tasks[i * chunk_size:(i + 1) * chunk_size]
            for i in range(math.ceil(len(tasks) / chunk_size))
        ]

        # Establish database connection
        db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )

        keys = []
        chains = {}
        for i, task_chunk in enumerate(task_chunks):
            with db_conn.begin(write=True, buffers=True) as txn:
                processed = Parallel(n_jobs=self.num_preprocess_jobs)(
                    task
                    for task in tqdm(task_chunk, desc=f"Chunk {i + 1}/{len(task_chunks)}")
                )
                stored = 0
                for data in processed:
                    if data is None:
                        continue
                    for key, value in data.items():
                        keys.append(key)
                        chains["_".join(key.split("_")[1:])] = value[1]
                        txn.put(key=key.encode(), value=pickle.dumps(value))
                        stored += 1
                print(f"[INFO] {stored} processed for chunk#{i + 1}")
        db_conn.close()

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)
        with open(self.chains_path, 'wb') as f:
            pickle.dump(chains, f)

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

        assert len(entry['mutations']) == torch.sum(data_dict_wt['aa'] != data_dict_mt['aa']),f"ID={data_dict_wt['id']},{len(entry['mutations'])},{torch.sum(data_dict_wt['aa'] != data_dict_mt['aa'])}"
        data_dict_wt['mut_flag'] = data_dict_mt['mut_flag'] = (data_dict_wt['aa'] != data_dict_mt['aa'])

        if self.transform is not None:
            data_dict_wt, idx_mask = self.transform(data_dict_wt)
            data_dict_mt, _ = self.transform(data_dict_mt, idx_mask)

        return {"wt": data_dict_wt,
                "mt": data_dict_mt
                }
def get_skempi_dataset(cfg):
    from rde.utils.transforms import get_transform
    return SkempiDataset_lmdb(
        csv_path=config.data.csv_path,
        pdb_wt_dir=config.data.pdb_wt_dir,
        pdb_mt_dir=config.data.pdb_mt_dir,
        cache_dir=config.data.cache_dir,
        num_cvfolds=self.num_cvfolds,
        cvfold_index=fold,
        transform=get_transform(config.data.transform)
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='../../data/SKEMPI2/SKEMPI2.csv')
    parser.add_argument('--pdb_wt_dir', type=str, default='../../data/SKEMPI2/SKEMPI2_cache/wildtype1')
    parser.add_argument('--pdb_mt_dir', type=str, default='../../data/SKEMPI2/SKEMPI2_cache/optimized1')
    parser.add_argument('--cache_dir', type=str, default='../../data/SKEMPI2/SKEMPI2_cache/entries_cache1')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = SkempiDataset_lmdb(
        csv_path = args.csv_path,
        pdb_wt_dir = args.pdb_wt_dir,
        pdb_mt_dir=args.pdb_mt_dir,
        cache_dir = args.cache_dir,
        split = 'train',
        num_cvfolds=2,
        cvfold_index=1,
        reset=args.reset,
    )
    print(dataset[0])
    print(len(dataset))