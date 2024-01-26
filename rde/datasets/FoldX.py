# -*- coding: utf-8 -*-
import sys
import numpy as np
import os, gc
import csv, glob
import os.path as path
import torch, pickle
from scipy.spatial import distance
import networkx as nx
import Bio.PDB
from matplotlib import pylab
import  pandas as pd

import scipy.sparse as sp
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def main():
    df = pd.read_csv('../../data/SKEMPI2/SKEMPI2.csv')
    for i, pdb_dict in df.iterrows():
        pdb_dir = '../../data/SKEMPI2/PDBs'
        workdir = '../../data/SKEMPI2/SKEMPI2_cache'
        pdbcode = pdb_dict["#Pdb"]
        if pdbcode == "1KBH":
            print("1KBH is locked!")
            continue
        pdb_id = pdb_dict["#Pdb"] + ".pdb"

        mutstr = pdb_dict["Mutation(s)_cleaned"]
        mut_list = pdb_dict["Mutation(s)_cleaned"].split(",")
        wild_list = []
        for mut in mut_list:
            wildname = list(mut)[0]
            chainid = list(mut)[1]
            resid = "".join(list(mut)[2:-1])
            mutname = list(mut)[-1]
            wild_list.append("".join([wildname, chainid, resid, wildname]))
        wildstr = ",".join(wild_list) + ";"
        mutstr = ",".join(mut_list) + ";"

        graph_out = os.path.join("../../data/SKEMPI2/SKEMPI2_cache/optimized", f"{str(i)}_{pdbcode}.pdb")
        os.system("mkdir -p {}".format(os.path.dirname(graph_out)))
        if os.path.exists(graph_out):
            print(f"{str(i)}_{pdbcode}.pdb exist!")
            continue

        print(f"generating {i}-th file")
        # build the wild-type file
        individual_file = os.path.join(workdir,'individual_list.txt')
        with open(individual_file, 'w') as f:
            cont = wildstr
            f.write(cont)

        comm = '../../data/SKEMPI2/FoldX --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(pdb_id, individual_file, workdir, pdb_dir, workdir)
        os.system(comm)

        wildtype_dir = os.path.join("{}/wildtype".format(workdir))
        if not os.path.exists(wildtype_dir):
            os.system("mkdir -p {}".format(wildtype_dir))
        os.system(f'mv {workdir}/{pdbcode}_1.pdb {wildtype_dir}/{str(i)}_{pdbcode}.pdb')

        # build the mutant file
        individual_file = os.path.join(workdir, 'individual_list.txt')
        with open(individual_file, 'w') as f:
            cont = mutstr
            f.write(cont)
        pdb_id = f"{str(i)}_{pdbcode}.pdb"
        pdb_dir = wildtype_dir
        comm = '../../data/SKEMPI2/FoldX --command=BuildModel --numberOfRuns=1 --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(pdb_id, individual_file, workdir, pdb_dir, workdir)
        os.system(comm)

        # energy optimization
        comm = '../../data/SKEMPI2/FoldX --command=Optimize --pdb={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(f"{str(i)}_" + pdbcode + "_1" + ".pdb", workdir, workdir, workdir)
        os.system(comm)

        Optimized_dir = os.path.join("{}/optimized1".format(workdir))
        if not os.path.exists(Optimized_dir):
            os.system("mkdir -p {}".format(Optimized_dir))
        os.system(f'mv {workdir}/Optimized_{str(i)}_{pdbcode}_1.pdb {Optimized_dir}/{str(i)}_{pdbcode}.pdb')

        os.system("rm {}/*.pdb".format(workdir))
        os.system("rm {}/*.fxout".format(workdir))


if __name__ == "__main__":
    main()