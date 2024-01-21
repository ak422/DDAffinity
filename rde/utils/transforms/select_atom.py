
from ._base import register_transform
import torch
import math
from rde.utils.protein.dihedral_chi import AA_TO_INDEX, ALL_ATOMS, ALL_ATOM_POSNS,ONE_TO_THREE, CHI_PI_PERIODIC_LIST
from rde.utils.protein.constants import restype_to_heavyatom_names as restype_name_to_atom15_names
from rde.utils.protein.constants import AA
import numpy as np


INDEX_TO_AA = {v:k for k,v in AA_TO_INDEX.items()}
residue_list = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "UNK"
    ]


chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
    'UNK':[]
}

atom_name_vocab = {
    "N": 0, "CA": 1, "C": 2, "O": 3, "CE3": 4, "CZ": 5, "SD": 6, "CD1": 7, "CB": 8,
    "NH1": 9, "OG1": 10, "CE1": 11, "OE1": 12, "CZ2": 13, "OH": 14, "CG": 15, "CZ3": 16,
    "NE": 17, "CH2": 18, "OD1": 19, "NH2": 20, "ND2": 21, "OG": 22, "CG2": 23, "OE2": 24,
    "CD2": 25, "ND1": 26, "NE2": 27, "NZ": 28, "CD": 29, "CE2": 30, "CE": 31, "OD2": 32,
    "SG": 33, "NE1": 34, "CG1": 35, "OXT": 36
}

max_num_allatoms = 37
max_num_heavyatoms = 15

restype_atom37_index_map = -torch.ones((len(residue_list), max_num_allatoms), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    resi_name3 = AA(resi_name3)
    for value, name in enumerate(restype_name_to_atom15_names[resi_name3]):
        if name in atom_name_vocab:
            restype_atom37_index_map[i_resi][atom_name_vocab[name]] = value  # 37表中对应于15表的哪个位置

atom15to37_index_map = -torch.ones((len(residue_list) , max_num_heavyatoms), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    resi_name3 = AA(resi_name3)
    for indx, atom in enumerate(restype_name_to_atom15_names[resi_name3]):
        if atom == '':
            continue
        atom15to37_index_map[i_resi][indx] = ALL_ATOM_POSNS[atom]


chi_atom37_index_map = -torch.ones((len(residue_list), 4, 4), dtype=torch.long)  # 处理残基padding
chi_atom15_index_map = -torch.ones((len(residue_list) , 4, 4), dtype=torch.long)  # 处理残基padding
for i_resi, resi_name3 in enumerate(residue_list):
    chi_angles_atoms_i = chi_angles_atoms[resi_name3]  # 二面角原子索引
    for j_chi, atoms in enumerate(chi_angles_atoms_i):
        for k_atom, atom in enumerate(atoms):
            chi_atom37_index_map[i_resi][j_chi][k_atom] = atom_name_vocab[atom]
            chi_atom15_index_map[i_resi][j_chi][k_atom] = restype_atom37_index_map[i_resi][atom_name_vocab[atom]]


@register_transform('select_atom')
class SelectAtom(object):
    def __init__(self, resolution):
        super().__init__()
        assert resolution in ('full', 'backbone', 'backbone+CB')
        self.resolution = resolution

    def __call__(self, data, dummy=None):
        if self.resolution == 'full':
            data['pos_atoms'] = data['pos_heavyatom'][:, :]
            data['mask_atoms'] = data['mask_heavyatom'][:, :]
            data['bfactor_atoms'] = data['bfactor_heavyatom'][:, :]
        elif self.resolution == 'backbone':
            data['pos_atoms'] = data['pos_heavyatom'][:, :, :4]
            data['mask_atoms'] = data['mask_heavyatom'][:, :, :4]
            data['bfactor_atoms'] = data['bfactor_heavyatom'][:, :, :4]
        elif self.resolution == 'backbone+CB':
            data['pos_atoms'] = data['pos_heavyatom'][:, :5]
            data['mask_atoms'] = data['mask_heavyatom'][:, :5]

        return data, dummy

@register_transform('set_chi')
class Set_chi(object):
    def __init__(self):
        super().__init__()
        # assert chi_id >=0 and chi_id <=1
        # self.chi_rotate_angles = chi_rotate_angles
    def _normalize_angles(self, angles):
        angles = angles % (2*np.pi)
        return torch.where(angles > np.pi, angles - 2*np.pi, angles)

    def rotate_side_chain(selft, pos_heavyatom, rotate_angles, residue_type):
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
    def __call__(self, data, setting = None):
        current_chi = data["chi"]
        residue_type = data["aa"]
        pos_heavyatom = data["pos_heavyatom"].clone()

        if setting == None:
            setting = torch.zeros_like(current_chi)
            # setting = setting + np.pi / 4
            setting = current_chi + np.pi / 4

        chi_to_rotate = setting - current_chi  # 旋转rotate_angles
        chi_to_rotate[torch.isnan(chi_to_rotate)] = 0
        new_pos_heavyatom = self.rotate_side_chain(pos_heavyatom, chi_to_rotate, residue_type)
        new_pos_allatom= transform_15to37(residue_type, new_pos_heavyatom)
        data["pos_heavyatom"] = new_pos_heavyatom
        data["pos_allatom"] = new_pos_allatom

        # new_mask_allatom = data["mask_allatom"]
        # output = get_chi(
        #                 new_pos_allatom,
        #                 new_mask_allatom,
        #                 residue_type,
        #                 )
        # new_chis = self._normalize_angles(output['dihedral'][:, :, 3:])
        # diff = (new_chis - current_chi).fmod(np.pi * 2) * data["chi_mask"]
        # test_mask = diff.isnan() | ((diff - np.pi / 4).abs() < 1e-4) | ((diff + np.pi * (8) / 4).abs() < 1e-4)
        # if not test_mask.all():
        #     print("chi setting!")
            # print(diff)
        # -------------------- #
        data = remove_by_chi(data, 1) # chi_id 从0开始计数 ,返回当前chi_mask, 返回当前chi及之前的原子
        data["chi"] = get_chis(data)   # 返回chi_mask有效的chi

        return data



def generate(batch, randomize=True):
        protein = batch['graph']
        if randomize:
            protein = rotamer.randomize(protein)

        schedule = self.schedule_1pi_periodic.reverse_t_schedule.to(self.device)
        for chi_id in tqdm(range(self.NUM_CHI_ANGLES), desc="Autoregressive generation"):
            for j in range(len(schedule) - 1):
                t = schedule[j]
                dt = schedule[j] - schedule[j + 1] if j + 1 < len(schedule) else 1
                chis = rotamer.get_chis(protein, protein.node_position)  # [num_residue, 4]

                # Predict score
                sigma = self.schedule_1pi_periodic.t_to_sigma(t).repeat(protein.batch_size)
                chi_protein = rotamer.remove_by_chi(protein, chi_id)  # 去掉 after_chi
                pred_score, _ = self.predict({
                    "graph": chi_protein,
                    "sigma": sigma,
                    "chi_id": chi_id
                })   # 仅用chi及之前的原子进行预测

                # Step backward
                chis = self.schedule_1pi_periodic.step(chis, pred_score, t, dt, chi_protein.chi_1pi_periodic_mask)
                chis = self.schedule_2pi_periodic.step(chis, pred_score, t, dt, chi_protein.chi_2pi_periodic_mask)
                protein = rotamer.set_chis(protein, chis)  # 仅预测当前chi
        return batch


@register_transform('set_all_chi')
class Set_all_chi(object):
    def __init__(self, chi_id, rotate_angles):
        super().__init__()
        assert resolution in ('full', 'backbone', 'backbone+CB')
        self.rotate_angles = rotate_angles
    def __call__(self, data, dummy=None):
        for i in  range(4):
            Set_chi(chi_id, rotate_angles)

