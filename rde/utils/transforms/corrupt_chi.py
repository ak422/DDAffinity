import torch
import numpy as np

from ._base import register_transform, _get_CB_positions
from rde.utils.protein.dihedral_chi import CHI_PI_PERIODIC_LIST
from rde.utils.transforms.noise import transform_15to37, get_dihedral


@register_transform('corrupt_chi_angle')
class CorruptChiAngle(object):

    def __init__(self, ratio_mask=0.1, noise_std_all=0.1, corrupt_dis=3.0, add_noise=True, maskable_flag_attr=None):
        super().__init__()
        self.ratio_mask = ratio_mask
        self.add_noise = add_noise
        self.noise_std_all = noise_std_all
        self.maskable_flag_attr = maskable_flag_attr
        self.corrupt_dis = corrupt_dis

    def _normalize_angles(self, angles):
        angles = angles % (2*np.pi)
        return torch.where(angles > np.pi, angles - 2*np.pi, angles)

    def _get_min_dist(self, data, center_idx):
        pos_beta_all = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])
        pos_beta_center = pos_beta_all[center_idx]
        cdist = torch.cdist(pos_beta_all, pos_beta_center)  # (L, K)
        min_dist = cdist.min(dim=1)[0]  # (L, )
        return min_dist

    def _get_noise_std(self, min_dist):
        return torch.clamp_min((-1/16) * min_dist + 1, 0)

    def _get_flip_prob(self, min_dist):
        return torch.where(
            min_dist <= self.corrupt_dis,
            torch.full_like(min_dist, 0.25),
            torch.zeros_like(min_dist,),
        )

    def _add_chi_gaussian_noise(self, chi, noise_std, chi_mask):
        """
        Args:
            chi: (L, 4)
            noise_std: (L, )
            chi_mask: (L, 4)
        """
        noise = torch.randn_like(chi) * noise_std[:, None] * chi_mask
        return self._normalize_angles(chi + noise)

    def _random_flip_chi(self, chi, flip_prob, chi_mask):
        """
        Args:
            chi: (L, 4)
            flip_prob: (L, )
            chi_mask: (L, 4)
        """
        delta = torch.where(
            torch.rand_like(chi) <= flip_prob[:, None],
            torch.full_like(chi, np.pi),
            torch.zeros_like(chi),
        ) * chi_mask
        return self._normalize_angles(chi + delta)

    def __call__(self, data, dummy=None):
        L = data['aa'].size(0)
        idx = torch.arange(0, L)
        if self.maskable_flag_attr is not None:
            flag = data[self.maskable_flag_attr]
            idx_mut = idx[flag]
        idx_mut = idx_mut.tolist()

        if dummy ==None:
            num_mask = max(int(self.ratio_mask * L), 1)
            idx = idx.tolist()
            np.random.shuffle(idx)
            idx_mask = idx[:num_mask]   # mask flag
        else:
            idx_mask = dummy

        # 对所有原子添加噪声
        # data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        mask_allatom = data["mask_allatom"]
        residue_type = data["aa"]
        _get_noise = lambda: ((torch.randn_like(data["pos_heavyatom"]) * self.noise_std_all) * data["mask_heavyatom"][:,:,None])
        data["pos_heavyatom"] = data["pos_heavyatom"] + _get_noise()
        data["pos_allatom"] = transform_15to37(residue_type.unsqueeze(0), data["pos_heavyatom"].unsqueeze(0))[0]

        output = get_dihedral(
            data["pos_allatom"],
            mask_allatom,
            residue_type,
        )
        data["phi"] = output['dihedral'][:, 1]
        data['psi'] = output['dihedral'][:, 2]
        data["chi"] = output['dihedral'][:, 3:]
        chi_is_ambiguous = data["chi"].new_tensor(
            CHI_PI_PERIODIC_LIST,
        )[residue_type, ...]

        mirror_torsion_angles = 1.0 - 2.0 * chi_is_ambiguous
        data["chi_alt"] = data["chi"] * mirror_torsion_angles

        # 添加二面角噪声
        min_dist = self._get_min_dist(data, idx_mask)  # 返回离ratio-mutation最近的距离
        noise_std = self._get_noise_std(min_dist)  #
        flip_prob = self._get_flip_prob(min_dist)  # 8埃范围以0.25概率

        chi_native = torch.where(
            torch.randn_like(data['chi']) > 0,
            data['chi'],
            data['chi_alt'],
        )   # (L, 4), randomly pick from chi and chi_alt 随机选择chi角和chi_alt角作为native
        data['chi_native'] = chi_native  # 包括chi & chi_alt
        chi_corrupt = chi_mut = chi_native.clone()
        chi_mask = data['chi_mask']

        if self.add_noise:
            chi_corrupt = self._add_chi_gaussian_noise(chi_corrupt, noise_std, chi_mask)
            chi_corrupt = self._random_flip_chi(chi_corrupt, flip_prob, chi_mask)  # 将chi角以概率加上pi
        chi_corrupt[idx_mask] = 0.0        # 去掉 ratio-mutation残基
        chi_mut[idx_mut] = 0.0           # 去掉 mutation残基

        corrupt_flag = torch.zeros(L, dtype=torch.bool)
        corrupt_flag[idx_mask] = True  #ratio-mutation残基置为1
        corrupt_flag[min_dist <= self.corrupt_dis] = True # 距离ratio-mutation残基8埃范围的也置为1

        masked_flag = torch.zeros(L, dtype=torch.bool)
        masked_flag[idx_mask] = True  # 仅包括ratio-mutation残基

        data['chi_corrupt'] = chi_corrupt           # 加入高斯噪声和８埃范围内加上ｐｉ的chi，且 去掉 ratio-mutation残基
        data['chi_mut'] = chi_mut           #  去掉 mutation残基
        data['chi_masked_flag'] = masked_flag  # 仅包括ratio-mutation残基
        data['chi_corrupt_flag'] = corrupt_flag    #  ratio-mutation残基 以及8埃范围内的残基
        # data[''mut_flag''] = mut_flag         # 仅包括mutation残基

        return data, idx_mask
