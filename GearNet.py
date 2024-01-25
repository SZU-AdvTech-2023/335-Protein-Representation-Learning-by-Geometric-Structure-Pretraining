from torchdrug import layers,data
from torchdrug.layers import geometry
from torchdrug import models
from torchdrug import transforms
import torch

#创建一个GearNet model
model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512,512,512,512], 
                        num_relation=7, edge_input_dim=59, num_angle_bin=8,
                        batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
state_dict = torch.load("/home/hzeng/p/Banana/mc_gearnet_edge.pth")  
model.load_state_dict(state_dict)
model.eval()

#创建一个GraphConstruction model
graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet"             )

#创建一个ProteinDataset
import logging
import warnings
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdmolops
import os
directory = "/home2/hzeng/anaconda3/envs/esm/src/results/pdb"
files = os.listdir(directory)
pdb_files = [os.path.join(directory, file) for file in files if file.endswith(".pdb")]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)# 设置日志级别为DEBUG
fh = logging.FileHandler('debug.log')# 创建一个handler，用于写入日志文件
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')# 定义handler的输出格式
fh.setFormatter(formatter)
logger.addHandler(fh)# 给logger添加handler
class MyProteinDataset(data.ProteinDataset):
    def __init__(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        super().__init__()
        self.targets={}
        self.load_pdbs(pdb_files, transform, lazy, verbose, **kwargs)
    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(pdb_file,sanitize=False) #sanitize=False: 不检查分子的化学有效性
                if not mol:
                    logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
                # try:
                #     rdmolops.SanitizeMol(mol) #修复分子
                # except Exception as e:
                #     logger.debug("Can't sanitize molecule from pdb file `%s`. Ignore this sample. Exception: %s" % (pdb_file, e))
                #     #continue
                protein = data.Protein.from_molecule(mol, **kwargs)
                if not protein:
                    logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_dense()#参考torchdrug.datasets.EnzymeCommission数据集，将张量转换为密集张量
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)  

#truncuate_transform = transforms.TruncateProtein(max_length=350, random=False) #截断蛋白质
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([protein_view_transform])
print("Loading PDB files...")
dataset = MyProteinDataset(pdb_files,transform=transform, atom_feature=None, bond_feature=None)
print(dataset)

save_path="/home/hzeng/GB_pred_rep/"
for i in range(len(dataset)):
    protein = dataset[i]['graph']
    filename=os.path.basename(dataset.pdb_files[i]).split(".")[0]
    _protein = data.Protein.pack([protein])
    protein_ = graph_construction_model(_protein)
    rep=model.forward(protein_,protein_.node_feature.float())
    torch.save(rep, save_path+filename+".pth")
    print(f"[{i}] {filename}.pth saved")