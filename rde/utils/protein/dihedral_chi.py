import os.path
import torch
# from protein_learning.common.data.data_types.protein import safe_load_sequence
from collections import  defaultdict
# from rde.utils.protein.pdb_utils import (
#     extract_atom_coords_n_mask_tensors,
#     extract_pdb_seq_from_pdb_file,
# )
from rde.utils.protein.helpers import disable_tf32
from rde.utils.protein.of_rigid_utils import Rigid
from easydict import EasyDict
import enum
from rde.utils.protein.constants import AA


BB_ATOMS = ["N", "CA", "C", "O"]
SC_ATOMS = [
    "CE3",    "CZ",    "SD",    "CD1",    "CB",    "NH1",    "OG1",    "CE1",    "OE1",    "CZ2",
    "OH",    "CG",    "CZ3",    "NE",    "CH2",    "OD1",    "NH2",    "ND2",    "OG",    "CG2",
    "OE2",   "CD2",    "ND1",   "NE2",    "NZ",    "CD",     "CE2",    "CE",     "OD2",    "SG",
    "NE1",   "CG1",    "OXT",
]
ALL_ATOMS = BB_ATOMS + SC_ATOMS

AA3LetterCode = [
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
    "UNK",
    "PAD"
]
AA1LetterCode = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y","#","-"]
THREE_TO_ONE = {three: one for three, one in zip(AA3LetterCode, AA1LetterCode)}
ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}
AA_TO_INDEX = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 4,
    "GLY": 5,
    "HIS": 6,
    "ILE": 7,
    "LYS": 8,
    "LEU": 9,
    "MET": 10,
    "ASN": 11,
    "PRO": 12,
    "GLN": 13,
    "ARG": 14,
    "SER": 15,
    "THR": 16,
    "VAL": 17,
    "TRP": 18,
    "TYR": 19,
    "UNK": 20,
    "PAD": 21,
}
AA_TO_INDEX.update({THREE_TO_ONE[aa]: v for aa, v in AA_TO_INDEX.items()})
ALL_ATOM_POSNS = {a: i for i, a in enumerate(ALL_ATOMS)}

AA_ALPHABET = "".join(
    ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y","#","-"]
)
RES_TYPES = [x for x in AA_ALPHABET]

CHI_ANGLES_MASK = {
    "ALA": [0.0, 0.0, 0.0, 0.0],  # ALA
    "ARG": [1.0, 1.0, 1.0, 1.0],  # ARG
    "ASN": [1.0, 1.0, 0.0, 0.0],  # ASN
    "ASP": [1.0, 1.0, 0.0, 0.0],  # ASP
    "CYS": [1.0, 0.0, 0.0, 0.0],  # CYS
    "GLN": [1.0, 1.0, 1.0, 0.0],  # GLN
    "GLU": [1.0, 1.0, 1.0, 0.0],  # GLU
    "GLY": [0.0, 0.0, 0.0, 0.0],  # GLY
    "HIS": [1.0, 1.0, 0.0, 0.0],  # HIS
    "ILE": [1.0, 1.0, 0.0, 0.0],  # ILE
    "LEU": [1.0, 1.0, 0.0, 0.0],  # LEU
    "LYS": [1.0, 1.0, 1.0, 1.0],  # LYS
    "MET": [1.0, 1.0, 1.0, 0.0],  # MET
    "PHE": [1.0, 1.0, 0.0, 0.0],  # PHE
    "PRO": [1.0, 1.0, 0.0, 0.0],  # PRO
    "SER": [1.0, 0.0, 0.0, 0.0],  # SER
    "THR": [1.0, 0.0, 0.0, 0.0],  # THR
    "TRP": [1.0, 1.0, 0.0, 0.0],  # TRP
    "TYR": [1.0, 1.0, 0.0, 0.0],  # TYR
    "VAL": [1.0, 0.0, 0.0, 0.0],  # VAL
    "UNK": [0.0, 0.0, 0.0, 0.0],  # UNK
    "PAD": [0.0, 0.0, 0.0, 0.0],  # PAD
}
def update_letters(x, is_three=True):
    mapping = THREE_TO_ONE if is_three else ONE_TO_THREE
    x.update({mapping[res]: value for res, value in x.items()})
    return x

CHI_ANGLES_MASK = update_letters(CHI_ANGLES_MASK)
CHI_ANGLES_MASK_LIST = [CHI_ANGLES_MASK[r] for r in RES_TYPES]

CHI_PI_PERIODIC = {
    "ALA": [0.0, 0.0, 0.0, 0.0],
    "ARG": [0.0, 0.0, 0.0, 0.0],
    "ASN": [0.0, 0.0, 0.0, 0.0],
    "ASP": [0.0, 1.0, 0.0, 0.0],
    "CYS": [0.0, 0.0, 0.0, 0.0],
    "GLN": [0.0, 0.0, 0.0, 0.0],
    "GLU": [0.0, 0.0, 1.0, 0.0],
    "GLY": [0.0, 0.0, 0.0, 0.0],
    "HIS": [0.0, 0.0, 0.0, 0.0],
    "ILE": [0.0, 0.0, 0.0, 0.0],
    "LEU": [0.0, 0.0, 0.0, 0.0],
    "LYS": [0.0, 0.0, 0.0, 0.0],
    "MET": [0.0, 0.0, 0.0, 0.0],
    "PHE": [0.0, 1.0, 0.0, 0.0],
    "PRO": [0.0, 0.0, 0.0, 0.0],
    "SER": [0.0, 0.0, 0.0, 0.0],
    "THR": [0.0, 0.0, 0.0, 0.0],
    "TRP": [0.0, 0.0, 0.0, 0.0],
    "TYR": [0.0, 1.0, 0.0, 0.0],
    "VAL": [0.0, 0.0, 0.0, 0.0],
    "UNK": [0.0, 0.0, 0.0, 0.0],
    "PAD": [0.0, 0.0, 0.0, 0.0],
}
CHI_PI_PERIODIC = update_letters(CHI_PI_PERIODIC)
CHI_PI_PERIODIC_LIST = [CHI_PI_PERIODIC[r] for r in RES_TYPES]

res_to_chi_atom_groups = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
    "UNK": [],
    "PAD": [],
}

non_standard_residue_substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

restype_to_heavyatom_names = {
    "A": ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    "R": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    '', 'OXT'],
    "N": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    '', 'OXT'],
    "D": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    '', 'OXT'],
    "C": ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    "Q": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    '', 'OXT'],
    "E": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    '', 'OXT'],
    "G": ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    "H": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    '', 'OXT'],
    "I": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    '', 'OXT'],
    "L": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    '', 'OXT'],
    "K": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    '', 'OXT'],
    "M": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    '', 'OXT'],
    "F": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    '', 'OXT'],
    "P": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    '', 'OXT'],
    "S": ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    "T": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    "W": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],
    "Y": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    '', 'OXT'],
    "V": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    "#": ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
    "-": ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
}

max_num_heavyatoms = 15
max_num_allatoms = 37
def map_tensor_fn_onto_iterable(data, fn):
    if isinstance(data,dict):
        return {k: map_tensor_fn_onto_iterable(v,fn) for k,v in data.items()}
    if isinstance(data,list):
        return [map_tensor_fn_onto_iterable(d,fn) for d in data]
    if isinstance(data,tuple):
        return tuple([map_tensor_fn_onto_iterable(d,fn) for d in data])
    return fn(data)

def _get_residue_heavyatom_info(pos, mask,  res_index, res):
    pos_allatom, mask_allatom = pos[res_index], mask[res_index]

    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    for idx, atom_name in enumerate(restype_to_heavyatom_names[res]):
        if atom_name == '': continue
        indice = ALL_ATOMS.index(atom_name)
        pos_heavyatom[idx] = torch.as_tensor(pos[res_index,indice], dtype=pos_heavyatom.dtype)
        mask_heavyatom[idx] = mask[res_index,indice]

    return pos_heavyatom, mask_heavyatom, pos_allatom, mask_allatom

def assess_sidechains(
        seq_wildtype,
        decoy_pdb_path,
        device="cpu",
        use_openfold_dihedral_calc = True,
):
    device = device if torch.cuda.is_available() else "cpu"
    decoy_protein = FromPDBAndSeq(
        pdb_path=decoy_pdb_path,
        seq=seq_wildtype,
        atom_tys=ALL_ATOMS,
    )

    stats = gather_target_stats(
        predicted_coords=decoy_protein[0],
        decoy_sequence=decoy_protein[2],
        decoy_atom_mask=decoy_protein[1],
        use_openfold_dihedral_calc=use_openfold_dihedral_calc,
    )
    stats["aa"] = seq_wildtype
    stats["pos_heavyatom"] = decoy_protein[0]
    stats["mask_heavyatom"] = decoy_protein[1]
    stats["res_ids"] = decoy_protein[3]

    return stats
#
# def FromPDBAndSeq(
#     pdb_path,
#     seq,
#     atom_tys,
#     remove_invalid_residues = False,
#     ignore_non_standard = True
# ):
#     coords, mask, res_ids, seq = extract_atom_coords_n_mask_tensors(
#         seq=seq,
#         pdb_path=pdb_path,
#         atom_tys=atom_tys,
#         remove_invalid_residues=remove_invalid_residues,
#         ignore_non_standard=ignore_non_standard,
#         return_res_ids=True,  # TODO
#     )
#     seq_encoding = torch.tensor([AA_TO_INDEX[r] for r in seq]).long()
#     return coords, mask, seq_encoding, res_ids



def get_chi_atom_indices():
    chi_atom_indices = []
    for residue_name in  RES_TYPES:
        residue_name =  ONE_TO_THREE[residue_name]
        residue_chi_groups = res_to_chi_atom_groups[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_groups:
            atom_indices.append([ALL_ATOM_POSNS[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)
    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.
    return chi_atom_indices

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def atom37_to_torsion_angles(
    protein,
    prefix="",
):
    device = "cpu" if protein[prefix + "aatype"].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device, enabled=False):
        N, CA, C, O = [ALL_ATOM_POSNS[x] for x in "N,CA,C,O".split(",")]
        aatype = protein[prefix + "aatype"]
        assert torch.max(aatype) < 21, f"{torch.max(aatype)}"
        all_atom_positions = protein[prefix + "all_atom_positions"]
        all_atom_mask = protein[prefix + "all_atom_mask"]

        aatype = torch.clamp(aatype, max=20)

        pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
        prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)  # 去除最后一个氨基酸
        next_all_atom_positions = torch.cat([all_atom_positions[..., 1:, :, :], pad], dim=-3)  # ak422

        pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
        prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)       # 去除最后一个氨基酸
        next_all_atom_mask = torch.cat([all_atom_mask[..., 1:, :], pad], dim=-2)  # ak422

        pre_omega_atom_pos = torch.cat(
            [prev_all_atom_positions[..., [CA, C], :], all_atom_positions[..., [N, CA], :]],
            dim=-2,
        )
        phi_atom_pos = torch.cat(
            [prev_all_atom_positions[..., [C], :], all_atom_positions[..., [N, CA, C], :]],
            dim=-2,
        )
        psi_atom_pos = torch.cat(
            # [all_atom_positions[..., [N, CA, C], :], all_atom_positions[..., [O], :]],
            [all_atom_positions[..., [N, CA, C], :], next_all_atom_positions[..., [N], :]],  # ak422
            dim=-2,
        )

        pre_omega_mask = torch.prod(prev_all_atom_mask[..., [CA, C]], dim=-1) * torch.prod(
            all_atom_mask[..., [N, CA]], dim=-1
        )
        phi_mask = prev_all_atom_mask[..., C] * torch.prod(
            all_atom_mask[..., [N, CA, C]], dim=-1, dtype=all_atom_mask.dtype
        )
        # psi_mask = torch.prod(all_atom_mask[..., [N, CA, C]], dim=-1, dtype=all_atom_mask.dtype) * all_atom_mask[..., O]
        psi_mask = torch.prod(all_atom_mask[..., [N, CA, C]], dim=-1, dtype=all_atom_mask.dtype) * next_all_atom_mask[..., N] # ak422

        chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)
        atom_indices = chi_atom_indices[..., aatype, :, :]
        chis_atom_pos = batched_gather(all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2]))
        chi_angles_mask = CHI_ANGLES_MASK_LIST
        chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
        chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)
        chis_mask = chi_angles_mask[aatype, :]
        chi_angle_atoms_mask = batched_gather(
            all_atom_mask,
            atom_indices,
            dim=-1,
            no_batch_dims=len(atom_indices.shape[:-2]),
        )
        chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype) # 若原子有缺失，则该二面角mask为0
        chis_mask = chis_mask * chi_angle_atoms_mask

        torsions_atom_pos = torch.cat(
            [
                pre_omega_atom_pos[..., None, :, :],
                phi_atom_pos[..., None, :, :],
                psi_atom_pos[..., None, :, :],
                chis_atom_pos,
            ],
            dim=-3,
        )

        torsion_angles_mask = torch.cat(
            [
                pre_omega_mask[..., None],
                phi_mask[..., None],
                psi_mask[..., None],
                chis_mask,
            ],
            dim=-1,
        )
        torsion_frames = Rigid.from_3_points(
            torsions_atom_pos[..., 1, :],
            torsions_atom_pos[..., 2, :],
            torsions_atom_pos[..., 0, :],
            eps=1e-8,
        )  # N：0, CA：1, C:2,
        fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

        torsion_angles_sin_cos = torch.stack([fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

        denom = torch.sqrt(
            torch.sum(
                torch.square(torsion_angles_sin_cos),
                dim=-1,
                dtype=torsion_angles_sin_cos.dtype,
                keepdims=True,
            )
            + 1e-10
        )
        torsion_angles_sin_cos = torsion_angles_sin_cos / denom

        torsion_angles_sin_cos[..., 2, :] = torsion_angles_sin_cos[..., 2, :] * -1  # 取反

        chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
            CHI_PI_PERIODIC_LIST,
        )[aatype, ...]

        mirror_torsion_angles = torch.cat(
            [
                all_atom_mask.new_ones(*aatype.shape, 3),
                1.0 - 2.0 * chi_is_ambiguous,
            ],
            dim=-1,
        )

        alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles[..., None]

    torsion_angles_sin_cos[..., 2, :] = torsion_angles_sin_cos[..., 2, :] * -1
    protein[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    protein[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos  # 考虑氨基酸侧链对称性
    protein[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return protein


def gather_target_stats(
    predicted_coords,
    decoy_atom_mask,
    decoy_sequence,
    use_openfold_dihedral_calc = False,
):
    output = dict()
    # if True:
    #     # 坐标扰动：分为backbone和sidechain
    #     predicted_coords = predicted_coords + 0.1 * torch.randn_like(predicted_coords)

    # number of CB neighbors in 10A ball
    # output["wildtype-centrality"] = compute_degree_centrality(predicted_coords.unsqueeze(0))[0]

    # compute dihedral stats
    output["dihedral"] = get_chi(
        predicted_coords=predicted_coords,
        decoy_atom_mask=decoy_atom_mask,
        decoy_sequence=decoy_sequence,
    )
    return output

# def parse_biopython_structure_perturb(entity, chains, unknown_threshold=1.0):
#     # chains = list(entity.keys())
#     # chains.sort()
#
#     data = EasyDict({
#         'chain_id': [], 'chain_nb': [],
#         'resseq': [],  'res_nb': [], 'residue_idx': [],
#         'aa': [],
#         'pos_heavyatom': [], 'mask_heavyatom': [],
#         'pos_allatom': [], 'mask_allatom': [],
#         'phi': [], 'phi_mask': [],
#         'psi': [], 'psi_mask': [],
#         'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [],
#     })
#     tensor_types = {
#         'chain_nb': torch.LongTensor,
#         'resseq': torch.LongTensor,
#         'res_nb': torch.LongTensor,
#         'residue_idx': torch.LongTensor,
#         'aa': torch.LongTensor,
#         'pos_heavyatom': torch.stack,
#         'mask_heavyatom': torch.stack,
#         'pos_allatom': torch.stack,
#         'mask_allatom': torch.stack,
#
#         'phi': torch.FloatTensor,
#         'phi_mask': torch.BoolTensor,
#         'psi': torch.FloatTensor,
#         'psi_mask': torch.BoolTensor,
#
#         'chi': torch.stack,
#         'chi_alt': torch.stack,
#         'chi_mask': torch.stack,
#         'chi_complete': torch.BoolTensor,
#     }
#
#     count_aa, count_unk = 0, 0
#     c = 1
#     l0 = 0
#     for i, chain in enumerate(chains):
#         seq_this = 0   # Renumbering residues
#         chain = chain.id
#         if chain not in entity.keys():
#             continue
#         residues = entity[chain][1]
#         for res_index, res in enumerate(residues):
#             # resname = res.get_resname()
#             if not AA.is_aa(res): continue
#             # if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
#             restype = AA(res)
#             count_aa += 1
#             if restype == AA.UNK:
#                 count_unk += 1
#                 continue
#
#             # Chain info
#             data.chain_id.append(chain)
#             data.chain_nb.append(i)
#
#             # Residue types
#             data.aa.append(restype) # Will be automatically cast to torch.long
#
#             # Heavy atoms
#             pos_heavyatom, mask_heavyatom, pos_allatom, mask_allatom = _get_residue_heavyatom_info(entity[chain][2],entity[chain][3],res_index, res)
#             data.pos_heavyatom.append(pos_heavyatom)
#             data.mask_heavyatom.append(mask_heavyatom)
#             data.pos_allatom.append(pos_allatom)
#             data.mask_allatom.append(mask_allatom)
#
#             # Backbone torsions
#             # omega-phi-psi
#             phi = entity[chain][0]["dihedral"][res_index][1]
#             phi_mask = entity[chain][0]["dihedral_mask"][res_index][1]
#             data.phi.append(phi)
#             data.phi_mask.append(phi_mask)
#             psi = entity[chain][0]["dihedral"][res_index][2]
#             psi_mask = entity[chain][0]["dihedral_mask"][res_index][2]
#             data.psi.append(psi)
#             data.psi_mask.append(psi_mask)
#
#             # Chi
#             chi = entity[chain][0]["dihedral"][res_index][3:7]
#             chi_alt = entity[chain][0]["dihedral_alt"][res_index][3:7]
#             chi_mask = entity[chain][0]["dihedral_mask"][res_index][3:7]
#             count_chi_angles = len(res_to_chi_atom_groups[ONE_TO_THREE[res]])
#             chi_complete = (count_chi_angles == sum(chi_mask))
#
#             data.chi.append(chi)
#             data.chi_alt.append(chi_alt)
#             data.chi_mask.append(chi_mask)
#             data.chi_complete.append(chi_complete)
#
#             # Sequential number
#             resseq_this = entity[chain][4][res_index].item()
#             # icode_this = res.get_id()[2]
#             if seq_this == 0:
#                 seq_this = 1
#             else:
#                 ALL_ATOMS_CA = ALL_ATOMS.index("CA")
#                 d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][ALL_ATOMS_CA] - data.pos_heavyatom[-1][ALL_ATOMS_CA], ord=2).item()
#                 if d_CA_CA <= 4.0:
#                     seq_this += 1
#                 else:
#                     d_resseq = resseq_this - data.resseq[-1]
#                     seq_this += max(2, d_resseq)
#
#             data.resseq.append(resseq_this)
#             # data.icode.append(icode_this)
#             data.res_nb.append(seq_this)
#             data.residue_idx.append(100 * (c - 1) + l0 + seq_this)
#
#         l0 += len(residues)
#         c += 1
#
#     if len(data.aa) == 0:
#         return None, None
#
#     if (count_unk / count_aa) >= unknown_threshold:
#         return None, None
#
#     seq_map = {}
#     for i, (chain_id, resseq) in enumerate(zip(data.chain_id, data.resseq)):
#         seq_map[(chain_id, resseq)] = i
#
#     for key, convert_fn in tensor_types.items():
#         data[key] = convert_fn(data[key])
#
#     return data, seq_map
#

def get_dihedral(pdb_path):
    sequences, chains = safe_load_sequence(None, pdb_path)
    dihedral_dict = defaultdict(list)
    for i, seq in enumerate(sequences):
        # N-termini don't have omega and phi
        # C-termini don't have psi
        res_level_stats = assess_sidechains(seq, pdb_path)
        dihedral_dict[chains[i]].extend([res_level_stats['dihedral']])
        dihedral_dict[chains[i]].extend([res_level_stats['aa']])
        dihedral_dict[chains[i]].extend([res_level_stats['pos_heavyatom']])
        dihedral_dict[chains[i]].extend([res_level_stats['mask_heavyatom']])
        dihedral_dict[chains[i]].extend([res_level_stats['res_ids']])
        # dihedral_dict[chains_wildtype[i]].extend([res_level_stats['wildtype-centrality']])
    return dihedral_dict
#
# if __name__ == '__main__':
#     wildtype_dir = os.listdir("./protein_learning/examples/wildtype_0_5_6")
#     wildtype_dir.sort(key=lambda x: int(x.split("_")[0]))
#     wildtype_pdb_pathes = [os.path.join("./protein_learning/examples/wildtype_0_5_6", inter) for inter in wildtype_dir]
#     #
#     # mutant_dir = os.listdir("./protein_learning/examples/mutant_0_5_6")
#     # mutant_dir.sort(key=lambda x: int(x.split("_")[0]))
#     # mutant_pdb_pathes = [os.path.join("./protein_learning/examples/mutant_0_5_6", inter) for inter in mutant_dir]
#
#     dihedral_info = {}
#     for wildtype_pdb_path in wildtype_pdb_pathes:
#         pdb_wt = os.path.basename(wildtype_pdb_path)
#         print(f"generating {pdb_wt}...")
#         dihedral_dict = get_dihedral(wildtype_pdb_path)
#         data, seq_map = parse_biopython_structure(dihedral_dict)

