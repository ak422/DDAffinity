import torch
import torch.nn as nn
from rde.modules.common.layers import AngularEncoding

class AAEmbedding(nn.Module):
    def __init__(self, feat_dim, infeat_dim):
        super(AAEmbedding, self).__init__()
        self.hydropathy = {'-': 0, '#': 0, "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "W": -0.9,
                           "G": -0.4, "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2, "N": -3.5, "D": -3.5,
                           "Q": -3.5, "E": -3.5, "K": -3.9, "R": -4.5}
        self.volume = {'-': 0, '#': 0, "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1, "P": 112.7, "N": 114.1,
                       "T": 116.1, "E": 138.4, "V": 140.0, "Q": 143.8, "H": 153.2, "M": 162.9, "I": 166.7, "L": 166.7,
                       "K": 168.6, "R": 173.4, "F": 189.9, "Y": 193.6, "W": 227.8}
        self.charge = {**{'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}, **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#-'}}
        self.polarity = {**{x: 1 for x in 'RNDQEHKSTY'}, **{x: 0 for x in "ACGILMFPWV#-"}}
        self.acceptor = {**{x: 1 for x in 'DENQHSTY'}, **{x: 0 for x in "RKWACGILMFPV#-"}}
        self.donor = {**{x: 1 for x in 'RKWNQHSTY'}, **{x: 0 for x in "DEACGILMFPV#-"}}

        alphabet = 'ACDEFGHIKLMNPQRSTVWY#-'

        self.embedding = torch.tensor([
            [self.hydropathy[alphabet[i]], self.volume[alphabet[i]] / 100, self.charge[alphabet[i]],
             self.polarity[alphabet[i]], self.acceptor[alphabet[i]], self.donor[alphabet[i]]]
            for i in range(len(alphabet))
        ])

        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view(1, 1, -1)  # [1, 1, K]
        D_expand = torch.unsqueeze(D, -1)  # [B, N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf(aa_vecs[:, :, 0], -4.5, 4.5, 0.1),
            self.to_rbf(aa_vecs[:, :, 1], 0, 2.2, 0.1),
            self.to_rbf(aa_vecs[:, :, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, :, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        B, N = x.size(0), x.size(1)
        aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs).to(x.device)
        return aa_vecs if raw else self.mlp(rbf_vecs)

    def to_rbf_(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).cuda()
        D_mu = D_mu.view(1, -1)  # [1, 1, K]
        D_expand = torch.unsqueeze(D, -1)  # [B, N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def soft_forward(self, x):
        aa_vecs = torch.matmul(x, self.embedding)
        rbf_vecs = torch.cat([
            self.to_rbf_(aa_vecs[:, 0], -4.5, 4.5, 0.1),
            self.to_rbf_(aa_vecs[:, 1], 0, 2.2, 0.1),
            self.to_rbf_(aa_vecs[:, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),
        ], dim=-1)
        return rbf_vecs

class PerResidueEncoder(nn.Module):
    def __init__(self, feat_dim, max_num_atoms,  max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types

        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim, padding_idx=21)
        self.dihed_embed = AngularEncoding()
        infeat_dim = feat_dim + self.dihed_embed.get_out_dim(6) # Phi, Psi, Chi1-4, mirror_Chi1-4

        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, phi, phi_mask, psi, psi_mask, chi, chi_mask, mask_residue):
        """
        Args:
            aa: (N, L)
            phi, phi_mask: (N, L)
            psi, psi_mask: (N, L)
            chi, chi_mask: (N, L, 4)
            mask_residue: (N, L)  # CA残基的mask
        """
        N, L = aa.size()

        # Amino acid identity features【氨基酸类型编码】
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        dihedral = torch.cat(
            [phi[..., None], psi[..., None], chi],
            dim=-1
        ) # (N, L, 6)
        dihedral_mask = torch.cat([
            phi_mask[..., None], psi_mask[..., None], chi_mask],
            dim=-1
        ) # (N, L, 6)

        # # Dihedral features 【骨架二面角和侧链二面角cat】
        dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 6, feat)
        dihedral_feat = dihedral_feat.reshape(N, L, -1)
        residue_feat = torch.cat([aa_feat, dihedral_feat], dim=-1)

        # Mix
        out_feat = self.mlp(residue_feat) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat

class PerResidueEncoder_backbone(nn.Module):
    def __init__(self, feat_dim, max_num_atoms,  max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types

        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim, padding_idx=21)
        self.dihed_embed = AngularEncoding()
        infeat_dim = feat_dim + self.dihed_embed.get_out_dim(2) # Phi, Psi

        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self,  aa, phi, phi_mask, psi, psi_mask, chi, chi_mask, mask_residue):
        """
        Args:
            aa: (N, L)
            phi, phi_mask: (N, L)
            psi, psi_mask: (N, L)
            chi, chi_mask: (N, L, 4)
            mask_residue: (N, L)  # CA残基的mask
        """
        N, L = aa.size()

        # Amino acid identity features【氨基酸类型编码】
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Dihedral features 【骨架二面角和侧链二面角cat】
        dihedral = torch.cat(
            [phi[..., None], psi[..., None]],
            dim=-1
        ) # (N, L, 2)
        dihedral_mask = torch.cat([
            phi_mask[..., None], psi_mask[..., None]],
            dim=-1
        ) # (N, L, 2)

        # # Dihedral features 【骨架二面角和侧链二面角cat】
        dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 2, feat)
        dihedral_feat = dihedral_feat.reshape(N, L, -1)
        residue_feat = torch.cat([aa_feat, dihedral_feat], dim=-1)

        # Mix
        out_feat = self.mlp(residue_feat) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat