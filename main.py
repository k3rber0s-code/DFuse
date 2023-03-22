from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import torch
from torch.utils.data import Dataset
import scipy.cluster.hierarchy
import numpy as np
import sys, argparse, bisect, re, os, fnmatch
import pickle, collections
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem.Fingerprints import FingerprintMols
import rdkit

import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

# See instructions here: https://github.com/gnina/models/tree/master/data/CrossDocked2020



class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True):

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

            self.data = {}

            lig_sections = np.where((np.diff(data['lig_mask'])) == 1)
            lig_sections += np.ones(len(lig_sections))
            lig_sections = lig_sections[0].astype(int)

            pocket_sections = np.where((np.diff(data['pocket_mask'])) == 1)
            pocket_sections += np.ones(len(pocket_sections))
            pocket_sections = pocket_sections[0].astype(int)


            for k, v in data.items():
                if "mask" in k:
                    continue
                elif "lig" in k:
                    self.data[k] = [torch.from_numpy(x) for x in np.split(v, lig_sections)]
                elif "pocket" in k:
                    self.data[k] = [torch.from_numpy(x) for x in np.split(v, pocket_sections)]
                else:
                    data[k] = v

class GCL(torch.nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, opt_features=1, est=False):
        super(GCL,self).__init__()
        
        self.est = est
        
        input_edge = 2 * input_nf # taking two feature vectors
        
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_edge + 1, hidden_nf ), #  + 1 for distance
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_nf,hidden_nf),
            torch.nn.SiLU()
        )
        
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_nf + hidden_nf, hidden_nf), # input_nf -> 1 feature vector + hidden_nf -> edge ?? nodes_att_dim
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_nf,output_nf),            
        )
        
        if self.est:
            self.estimation_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_nf,1),
                torch.nn.Sigmoid()
            )
            
        def edge_model(self, node1, node2, attributes): # calculating mij
            output = torch.cat([node1 , node2, attributes], dim=1)
            
            m_ij = self.edge_mlp(output)
            
            if self.est:
                edge_est = self.estimation_mlp(m_ij)
                output = m_ij * edge_est
            else:
                output = m_ij
                
            return output, m_ij
        
        def node_model(self, node, edge_index, node_att, edge_att ): # node update
            row, col = edge_index
            agg = unsorted_segment_sum(edge_att,row,num_segments=node.size(0), normalization_factor=normalization_factor,agg_method=agg_method)
            
            if node_att is not None:
                agg = torch.cat([node,agg,node_att], dim=1)
            else:
                agg = torch.cat([node,agg], dim=1)
                
            out = node  + self.node_mlp(agg)
            return out, agg
        
        #def forward(self,):
        
def unsorted_segment_sum(data, segment_ids, num_segments, agg_method, normalization_factor):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape,0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1,data.size(1))
    result.scatter_add_(0, segment_ids,data)
    if agg_method == 'sum':
        result = result / normalization_factor
        
    elif agg_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0,segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
        
    return result




dataset = ProcessedLigandPocketDataset(npz_path="./DiffSBDD/test.npz").data

for (k,v) in dataset.items():
    dataset[k] = v[:10]
    
print(dataset)
'''  
test = [dataset[k][0] for k in dataset ]   
H = nx.DiGraph()
node_names = list()
for i in range(len(test[0])):
    H.add_node(f"Lig_{i}", coord=test[0][i], atom=test[1][i])
    node_names.append(f"Lig_{i}")

for i in range(len(test[2])):
    H.add_node(f"Prot_{i}", coord=test[2][i], atom=test[3][i])
    node_names.append(f"Prot_{i}")
    
edges = combinations(node_names,2)

H.add_edges_from(edges)
G = H.to_undirected()

nx.draw(G)
plt.show()
'''
print()