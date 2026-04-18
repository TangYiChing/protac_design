import os
import json
import torch
import tempfile
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import categorical_edge_match, categorical_node_match

ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P':8}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P':15}

# One-hot atom types
NUMBER_OF_ATOM_TYPES = len(ATOM2IDX)

class PROTAC:
    """
    Process PROTAC-DB's PROTAC Dataset
    :param protac_json: json file of protacdb dataset, shared column: row_idx
    """
    def __init__(self, protac_json):
        self.protac_data = load_json(protac_json)
        
        # temp
        PID = os.getpid()
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"tmp_sdfs_{PID}")
        os.makedirs(self.temp_dir, exist_ok=True)

    def create_data(self):
        dataset = []
        fout_path = self.temp_dir
        for ridx, data_dict in self.protac_data.items():
            keys = data_dict.keys()
            protac_id = ridx
            protac_smi = data_dict["protac_smiles"]
            linker_smi = data_dict["linker_smiles"]

            if "protac_sdf" in keys and "linker_sdf" in keys:
                protac_sdf = data_dict["protac_sdf"]
                linker_sdf = data_dict["linker_sdf"]
            else:
                protac_sdf = f"{fout_path}/{ridx}_protac.sdf"
                linker_sdf = f"{fout_path}/{ridx}_linker.sdf"
                smi2sdffile(protac_smi, protac_sdf)
                smi2sdffile(linker_smi, linker_sdf)

            if os.path.isfile(protac_sdf) and os.path.isfile(linker_sdf):
                feature = self.sdf_to_3d_features(protac_sdf, linker_sdf, protac_id)
                if feature is not None:
                    dataset.append(feature)

        return dataset
    
    def sdf_to_3d_features(self, protac_sdf, linker_sdf, protac_id):
        # get graphs
        G = sdf2nx(protac_sdf)
        G_linker = sdf2nx(linker_sdf)

        # mapping
        maps, anchors = get_map_ids_from_nx(G, G_linker)
        if len(maps) ==1:
            n = len(G.nodes)
            n0 = G.nodes
            n1 = maps[0] # linker
            n2 = list(set(n0) - set(n1)) # fragment
            positions = []
            one_hot = [] 
            charges = []
            in_anchors = []
            fragment_mask = []
            linker_mask = []

            for ligand_atom in n2:
                positions.append(G.nodes[ligand_atom]['positions'])
                fragment_mask.append(1.)
                linker_mask.append(0.)

                tmp = [0.]*NUMBER_OF_ATOM_TYPES
                tmp[ATOM2IDX[G.nodes[ligand_atom]['element']]]=1.
                one_hot.append(tmp)
                charges.append(CHARGES[G.nodes[ligand_atom]['element']])
                if ligand_atom in anchors[0]:
                    in_anchors.append(1.)
                else:
                    in_anchors.append(0.)
            
            for linker_atom in n1:
                positions.append(G.nodes[linker_atom]['positions'])
                fragment_mask.append(0.)
                linker_mask.append(1.)

                tmp = [0.]*NUMBER_OF_ATOM_TYPES
                tmp[ATOM2IDX[G.nodes[linker_atom]['element']]]=1.
                one_hot.append(tmp)
                charges.append(CHARGES[G.nodes[linker_atom]['element']])


            # get features
            features = {
                "uuid": protac_id,
                "name": protac_id,
                "positions": torch.tensor(positions),
                "one_hot": torch.tensor(one_hot),
                "charges": torch.tensor(charges),
                "anchors": torch.tensor(in_anchors),
                "linker_mask": torch.tensor(linker_mask),
                "fragment_mask": torch.tensor(fragment_mask),
                "num_atoms":n
            }
            return features
        else:
            return None

        

def load_json(json_path):
    with open(json_path) as fin:
        data = json.load(fin)
    return data

def smi2sdffile(smi, sdf_path):
    m1 = Chem.MolFromSmiles(smi)
    Chem.Kekulize(m1)
    m2 = Chem.AddHs(m1)
    tmp = AllChem.EmbedMolecule(m2, useRandomCoords=True)
    if tmp<0:
        print(f'{sdf_path} failed')
    else:
        AllChem.MMFFOptimizeMolecule(m2) 
        m3 = Chem.RemoveHs(m2)
        w =  Chem.SDWriter(sdf_path)
        w.SetKekulize(False)
        w.write(m3)
        w.close()

def sdf2nx(sdf_path):
    with open(sdf_path) as f:
        sdf = f.readlines()
    G = nx.Graph()
    nodes_id = 0
    for lines in range(len(sdf)):
        tmp = sdf[lines].split()
        if len(tmp) == 16:
            G.add_node(
                nodes_id, 
                element=tmp[3], 
                positions=[float(_) for _ in tmp[:3]]
            )
            nodes_id += 1
        if len(tmp) == 4:
            G.add_edge(int(tmp[0])-1, int(tmp[1])-1, type=tmp[2])
    return G

def get_map_ids_from_nx(G, G_linker):
    nm = categorical_node_match("element",['C','N','O','S','F','Cl','Br','I'])
    em = categorical_edge_match("type",['1','2','3','4','5'])
    GM = isomorphism.GraphMatcher(G, G_linker,node_match=nm, edge_match=em)
    maps = []
    anchors = []
    for i in GM.subgraph_isomorphisms_iter():
        n0 = G.nodes
        n1 = list(i.keys()) # linker
        n1.sort()
        n2 = list(set(n0) - set(n1)) # ligand
        e0 = G.edges
        e1 = G.subgraph(n1).edges
        e2 = G.subgraph(n2).edges
        if len(e1) + len(e2) + 2 == len(e0) and (n1 not in maps):
            maps.append(n1)
            lost_e = list(set(e0) - set(e1)-set(e2))
            anchor = []
            for j in lost_e:
                if j[0] not in n1: anchor.append(j[0])
                if j[1] not in n1: anchor.append(j[1])
            anchors.append(set(anchor))
    return maps, anchors