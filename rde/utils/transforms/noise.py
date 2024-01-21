import torch
import numpy as np
import copy

from ._base import register_transform, _get_CB_positions
from rde.utils.transforms.select_atom import chi_atom15_index_map,atom15to37_index_map
from rde.utils.protein.dihedral_chi import  atom37_to_torsion_angles, AA_TO_INDEX, ALL_ATOMS, ALL_ATOM_POSNS,ONE_TO_THREE, CHI_PI_PERIODIC_LIST


max_num_allatoms = 37
max_num_heavyatoms = 15


def get_chis(data, chi_id):
    current_chi_corrupt = data["chi_corrupt"] * data["chi_mask"]   #
    current_chi_mask = data["chi_mask"].clone()

    current_chi_mask[:, :, :chi_id] = 0
    current_chi_mask[:, :, chi_id + 1:] = 0

    # pos_allatom = data["pos_allatom"]
    # mask_allatom = data["mask_allatom"]
    # residue_type = data["aa"]
    # current_chi_dict = get_dihedral(
    #     pos_allatom,
    #     mask_allatom,
    #     residue_type,
    # )
    # current_chi = current_chi_dict["dihedral"][:,:,3:] * data["chi_mask"]      # 仅当前chi有效
    #
    # chi_is_ambiguous = current_chi.new_tensor(
    #     CHI_PI_PERIODIC_LIST,
    # )[residue_type, ...]
    #
    # mirror_torsion_angles = 1.0 - 2.0 * chi_is_ambiguous
    # current_chi_alt = current_chi * mirror_torsion_angles

    return current_chi_corrupt, current_chi_mask

def remove_by_chi(data, chi_id):
    new_protein = copy.deepcopy(data)
    # residue_type = new_protein["aa"]
    # mask_heavyatom = new_protein["mask_heavyatom"]
    # mask_allatom = torch.zeros([residue_type.shape[0], residue_type.shape[1], max_num_allatoms, ],
    #                            dtype=torch.bool, device=residue_type.device)
    # pos_allatom = torch.zeros([residue_type.shape[0], residue_type.shape[1], max_num_allatoms, 3], dtype=torch.float,
    #                           device=residue_type.device)
    # 仅预测当前chi
    if chi_id == 3:
        return new_protein
    elif chi_id >=0 and chi_id<3:
        # new_protein["chi_mask"][:, :, :chi_id] = 0
        new_protein["chi_mask"][:, :, chi_id + 1:] = 0
    # new_protein["chi_corrupt"] = new_protein["chi_corrupt"]  * new_protein["chi_mask"]

    # # 表15的索引
    # chi_atom15_index = chi_atom15_index_map.to(residue_type.device)[residue_type]
    # atom_4 = chi_atom15_index[:, :, chi_id + 1, -1]  # after_chi 的第4原子作为分界点
    # atom15index = torch.arange(max_num_heavyatoms, device=residue_type.device)[None, None, :]
    # mask_atom = (atom15index >= atom_4[:,:,None]) & (atom_4[:,:,None] != -1)
    # # 仅选择after_chi 的第4原子之前的原子有效
    # mask_heavyatom = (~mask_atom & mask_heavyatom).unsqueeze(-1)
    #
    # zeros_pos_heavyatom = torch.zeros_like(mask_heavyatom, device=residue_type.device, dtype=torch.long)
    # new_protein["pos_heavyatom"] = torch.where(mask_heavyatom, new_protein["pos_heavyatom"], zeros_pos_heavyatom)
    # mask_heavyatom = mask_heavyatom.squeeze(-1)
    # new_protein["mask_heavyatom"] = mask_heavyatom
    #
    # indx = atom15to37_index_map[residue_type].to(residue_type.device)
    # for i in range(residue_type.shape[0]):
    #     for j in range(residue_type.shape[1]):
    #         pos_allatom[i, j, indx[i, j,:]] = new_protein["pos_heavyatom"][i, j, :]
    #         mask_allatom[i, j, indx[i, j,:]] = mask_heavyatom[i, j, :]
    #
    # new_protein["pos_allatom"] = pos_allatom
    # new_protein["mask_allatom"] = mask_allatom

    return new_protein

def rotate_side_chain( pos_heavyatom, rotate_angles, residue_type):
    # Rodrigues 旋转公式
    node_position15 = pos_heavyatom
    chi_atom15_index = chi_atom15_index_map.to(pos_heavyatom.device)[residue_type]  # (num_residue, 4, 4) 0~13
    chi_atom15_mask = chi_atom15_index != -1
    chi_atom15_index[~chi_atom15_mask] = 0

    for i in range(4):
        atom_1, atom_2, atom_3, atom_4 = chi_atom15_index[:, :, i, :].unbind(-1)  # (batch, num_residue, )  原子索引[0,14)
        atom_2_position = torch.gather(node_position15, -2,
                                       atom_2[:, :, None, None].expand(-1, -1, -1, 3))  # (num_residue, 1, 3)
        atom_3_position = torch.gather(node_position15, -2,
                                       atom_3[:, :, None, None].expand(-1, -1, -1, 3))  # (num_residue, 1, 3)
        k = atom_3_position - atom_2_position
        k_normalize = (k / (k.norm(dim=-1, keepdim=True) + 1e-10))
        rotate_angle = rotate_angles[:, :,  i, None , None]  # 选择第0个角度

        # Rotate all subsequent atoms by the rotation angle
        rotate_atoms_position = node_position15 - atom_2_position  # (num_residue, 14, 3)
        p_parallel = (rotate_atoms_position * k_normalize).sum(dim=-1, keepdim=True) \
                     * k_normalize

        normal_vector = torch.cross(k_normalize.expand(-1, -1, max_num_heavyatoms, -1), rotate_atoms_position, dim=-1)
        transformed_atoms_position = rotate_atoms_position * rotate_angle.cos() + \
                                     (1 - rotate_angle.cos()) * p_parallel + \
                                     normal_vector * rotate_angle.sin() + \
                                     atom_2_position  # (num_residue, 14, 3)

        assert not transformed_atoms_position.isnan().any()
        chi_mask = chi_atom15_mask[:, :, i, :].all(dim=-1, keepdim=True)  # (num_residue, 1)
        atom_mask = torch.arange(max_num_heavyatoms, device=pos_heavyatom.device)[None, None,:] >= atom_4[:, :, None]  # (num_residue, 14) 每个chi角之后的原子需要旋转
        mask = (atom_mask & chi_mask).unsqueeze(-1).expand_as(node_position15)
        node_position15[mask] = transformed_atoms_position[mask]

    return node_position15

def transform_15to37(residue_type, node_position15):
    pos_allatom = torch.zeros([node_position15.shape[0], node_position15.shape[1], max_num_allatoms, 3], dtype=torch.float, device=residue_type.device)

    indx = atom15to37_index_map[residue_type].to(residue_type.device)
    for i in range(residue_type.shape[0]):
        for j in range(residue_type.shape[1]):
            pos_allatom[i, j, indx[i, j,:]] = node_position15[i, j,:]

    return pos_allatom

def get_dihedral(
    predicted_coords,
    decoy_atom_mask,
    decoy_sequence,
):
    out = dict()
    predicted_coords = predicted_coords.unsqueeze(0)
    decoy_atom_mask = decoy_atom_mask.unsqueeze(0)
    decoy_sequence = decoy_sequence.unsqueeze(0)
    dihedral_info = atom37_to_torsion_angles(dict(aatype=decoy_sequence, all_atom_positions=predicted_coords, all_atom_mask=decoy_atom_mask)
    )

    pred_angle = torch.atan2(*dihedral_info["torsion_angles_sin_cos"].unbind(-1))
    alt_pred_angle = torch.atan2(*dihedral_info["alt_torsion_angles_sin_cos"].unbind(-1))

    out["dihedral"] = pred_angle[0]
    out["dihedral_alt"] = alt_pred_angle[0]
    out["dihedral_mask"] = dihedral_info["torsion_angles_mask"].bool()[0]

    return out

def set_chis(data, chi_id, setting = None):
    # # chi_corrupt: 仅选择当前及之前的chi_corrupt
    # current_chi = data["chi"]
    # residue_type = data["aa"]
    # pos_heavyatom = data["pos_heavyatom"]
    #
    # if setting == None:
    #     setting = torch.zeros_like(current_chi)
    #     setting = current_chi + np.pi / 4
    #
    # chi_to_rotate = (setting - current_chi) * data["chi_mask"]  # 旋转rotate_angles
    # chi_to_rotate[torch.isnan(chi_to_rotate)] = 0
    # new_pos_heavyatom = rotate_side_chain(pos_heavyatom, chi_to_rotate, residue_type)
    # new_pos_allatom= transform_15to37(residue_type, new_pos_heavyatom)
    # data["pos_heavyatom"] = new_pos_heavyatom
    # data["pos_allatom"] = new_pos_allatom
    #
    # new_mask_allatom = data["mask_allatom"]
    # output = get_dihedral(
    #                 new_pos_allatom,
    #                 new_mask_allatom,
    #                 residue_type,
    #                 )
    # data["chi_corrupt"] = output['dihedral'][:, :, 3:] * data["chi_mask"]

    # # diff = (new_chis - current_chi).fmod(np.pi * 2) * data["chi_mask"]
    # # test_mask = diff.isnan() | ((diff - np.pi / 4).abs() < 1e-4) | ((diff + np.pi * (8) / 4).abs() < 1e-4)
    # # if not test_mask.all():
    # #     print("chi setting!")
    #     # print(diff)
    # # -------------------- #
    chi_corrupt = data["chi_corrupt"].clone()
    chi_corrupt = setting
    chi_corrupt = setting * data["chi_mask"]

    return chi_corrupt[:,:,chi_id]



@register_transform('add_atom_noise')
class AddAtomNoise(object):

    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

    def __call__(self, data, dummy=None):
        protein = copy.deepcopy(data)
        # pos_atoms = data['pos_atoms']   # (L, A, 3)
        # mask_atoms = data['mask_atoms'] # (L, A)
        # noise = (torch.randn_like(pos_atoms) * self.noise_std) * mask_atoms[:, :, None]
        # pos_noisy = pos_atoms + noise
        # data['pos_atoms'] = pos_noisy
        # return data

        # 对所有原子添加噪声
        # data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        mask_allatom = protein["mask_allatom"]
        residue_type = protein["aa"]
        _get_noise = lambda: (
                    (torch.randn_like(protein["pos_heavyatom"]) * self.noise_std) * protein["mask_heavyatom"][:, :, None])
        protein["pos_heavyatom"] = protein["pos_heavyatom"] + _get_noise()
        protein["pos_allatom"] = transform_15to37(residue_type.unsqueeze(0), protein["pos_heavyatom"].unsqueeze(0))[0]

        output = get_dihedral(
            protein["pos_allatom"],
            mask_allatom,
            residue_type,
        )
        protein["phi"] = output['dihedral'][:, 1]
        protein['psi'] = output['dihedral'][:, 2]
        protein["chi"] = output['dihedral'][:, 3:]
        chi_is_ambiguous = protein["chi"].new_tensor(
            CHI_PI_PERIODIC_LIST,
        )[residue_type, ...]

        mirror_torsion_angles = 1.0 - 2.0 * chi_is_ambiguous
        protein["chi_alt"] = protein["chi"] * mirror_torsion_angles

        return protein, dummy

@register_transform('select_corrupt_chi')
class ChiSelection(object):
    # batch操作
    def __init__(self, noise_std=None):
        super().__init__()
        self.noise_std = noise_std

    def _normalize_angles(self, angles):
        angles = angles % (2*np.pi)
        return torch.where(angles > np.pi, angles - 2*np.pi, angles)

    def __call__(self, data,  chi_id=None):

        if chi_id is not None:
            # 将after_chi_mask置0
            data = remove_by_chi(data, chi_id)  # 仅修改chi_mask

        # current_chi_corrupt: 当前及之前的chi，current_chi_mask：仅当前chi有效
        current_chi_corrupt, current_chi_mask = get_chis(data, chi_id)  # all_chi_corrupt  # (L, 4)
        # if self.noise_std == None:
        #     _get_noise = lambda: torch.zeros_like(current_chi_corrupt, device=current_chi_corrupt.device, dtype=torch.long)
        # else:
        #     _get_noise = lambda: ((torch.randn_like(current_chi_corrupt) * self.noise_std) * current_chi_mask)
        # current_chi_corrupt = self._normalize_angles( current_chi_corrupt + _get_noise() )
        # 仅选择当前及之前的chi_corrupt
        data = set_chis(data, chi_id, current_chi_corrupt)
        return data
