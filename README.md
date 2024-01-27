# DDAffinity-network
<img src="./assets/cover.png" alt="cover" style="width:50%;" />

DDAffinity: Predicting the changes in binding affinity of multiple point mutations using protein three-dimensional structure

[[Github]](https://github.com/ak422/DDAffinity)

## Install

### Environment

```bash
conda env create -f env.yml -n DDAffinity
conda activate DDAffinity
```

The default PyTorch version is 1.12.1 and cudatoolkit version is 11.3. They can be changed in [`env.yml`](./env.yml).

## Preparation of processed dataset

We generated all protein mutant complex PDB data and wild-type complex PDB data from PDBs file [data/SKEMPI2/PDBs](https://drive.google.com/file/d/1SQTxpGr3P9hFhzmPCGIlAf0ggBSVoDVi/view?usp=drive_link), rde/datasets/PDB_generate.py, [data/SKEMPI2/SKEMPI2.csv](https://drive.google.com/file/d/15KHjAh_wIcoEbEmS5AHslewJHArgBvIc/view?usp=drive_link), and [FoldX](https://foldxsuite.crg.eu/) tool. Then we use rde/datasets/skempi_parallel.py to transform the PDB files of wild-type and mutant complexes into processed dataset [SKEMPI2_cache](https://drive.google.com/file/d/1p2ky9I8CwbCErGF0fw9jrAvrFkV95ZMe/view?usp=drive_link).

```bash
python PBD_generate.py 
python skempi_parallel.py --reset
```

### Datasets

| Dataset                                      | Download Script                                   | Processed Dataset                                                                                                     |
|----------------------------------------------| ------------------------------------------------- |-----------------------------------------------------------------------------------------------------------------------|
| [SKEMPI v2](https://life.bsc.es/pid/skempi2) | [`data/get_skempi_v2.sh`](./data/get_skempi_v2.sh) | [`data/SKEMPI2/SKEMPI2_cache`](https://drive.google.com/file/d/1p2ky9I8CwbCErGF0fw9jrAvrFkV95ZMe/view?usp=drive_link) |
| [SKEMPI2.csv](https://drive.google.com/file/d/15KHjAh_wIcoEbEmS5AHslewJHArgBvIc/view?usp=drive_link)                                       | —                                                 | [SKEMPI2_cache](https://drive.google.com/file/d/1p2ky9I8CwbCErGF0fw9jrAvrFkV95ZMe/view?usp=drive_link)                |
| [M1707.csv](https://drive.google.com/file/d/1JdXZ_DaYC0AusU6gpgQu0KWc-7W4ilf6/view?usp=drive_link)                                       | —                                                 | [M1707_cache](https://drive.google.com/file/d/1Ff6E59qoX3ahvRWWlEssW-KyEbG3QrIC/view?usp=drive_link)                  |
| [S1131.csv](https://drive.google.com/file/d/1yq8C2MVVBa51Zh1icwsM2cPWAE_Z_m9C/view?usp=drive_link)                                        | —                                                 | [S1131_cache](https://drive.google.com/file/d/1kQYDqB5hQF48hw6ebYZticuefSUmor9D/view?usp=drive_link)                  |
| [M1340.csv](https://drive.google.com/file/d/1dX_RUbxNg5NEp1QzEF7-YAqGn-vHiwJv/view?usp=drive_link)                                       | —                                                 | [M1340_cache](https://drive.google.com/file/d/1dQ4rJFnuEOjUmw53xgpg0NQMklyEWHtU/view?usp=drive_link)                        |
| [M595.csv](https://drive.google.com/file/d/1RarKcz3eya0gSHUX3C4xPtzLYnu3jbBH/view?usp=drive_link)                                        | —                                                 | [M595_cache](https://drive.google.com/file/d/1INg4pKe5d-PfzuPS74KqZCazeLG2-I7l/view?usp=drive_link)                   |
| [S494.csv](https://drive.google.com/file/d/1DiFiYAoWZNP-x9YxfleBpa7WfUladeuY/view?usp=drive_link)                                         | —                                                 | [S494_cache](https://drive.google.com/file/d/12zHpyKEkVi7Ou4zI6jGzY9hlVUzQt35G/view?usp=drive_link)                   |
| [S285.csv](https://drive.google.com/file/d/1BqHuKV35ybQZxO_PXzlwYlJhQcwZZGdv/view?usp=drive_link)                                         | —                                                 | [S285_cache](https://drive.google.com/file/d/1-ia2Qpy3VbOSbi6Ay5wQAjvVAgnkrXJk/view?usp=drive_link)                   |

### Trained Weights
The overall SKEMPI2 trained weights is located in:
[DDAffinity](https://drive.google.com/file/d/1JLdHrKkwWLTsiNBaH8-x9qnEpfp9q-73/view?usp=drive_link)

The M1340 trained weights is located in:
[M1340](https://drive.google.com/file/d/12_nh2Z1PA16Icm1H1dh_ndafPnGMdm1Z/view?usp=drive_link)

## Usage

### Evaluate DDAffinity

```bash
python test_DDAffinity.py ./configs/train/mpnn_ddg.yml --device cuda:0
```

### Blind testing: non-redundant blind testing on the multiple point mutation dataset M595

```bash
python case_study.py ./configs/inference/blind_testing.yml --device cuda:0
```

### Case Study 1: Predict Mutation Effects for SARS-CoV-2 RBD

```bash
python case_study.py ./configs/inference/case_study_1.yml --device cuda:0
```

### Case Study 2: Human Antibody Optimization

```bash
python case_study.py ./configs/inference/case_study_2.yml --device cuda:0
```

### Train DDAffinity

```bash
python train_DDAffinity.py ./configs/train/mpnn_ddg.yml --num_cvfolds 10 --device cuda:0
```
# Acknowledgements
We acknowledge that parts of our code is adapted from [Rotamer Density Estimator (RDE)](https://github.com/luost26/RDE-PPI). Thanks to the authors for sharing their codes. 