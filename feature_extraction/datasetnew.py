from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset, Data
# from feature_extraction.get_geo_features import get_geo_features_new, get_geo_features_test1
from torch_geometric.transforms import AddLaplacianEigenvectorPE

    
class RNAGraphDatasetNew(InMemoryDataset):
    def __init__(self, root,
                 pdb_dir=None, fasta_dir=None, label_file_path=None,
                 topk=None, rna_emb_dir = None,rna_ss_dir=None, asa_dir = None, transform=None, pre_transform=None, force_reload=False):
        self.pdb_dir   = Path(pdb_dir) if pdb_dir else None
        self.fasta_dir = Path(fasta_dir) if fasta_dir else None
        self.label_file_path = label_file_path
        self.topk = topk
        self.rna_emb_dir = rna_emb_dir
        self.rna_ss_dir = rna_ss_dir
        self.asa_path = asa_dir

        
        super().__init__(root, transform, pre_transform,force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["placeholder.txt"]  # không dùng raw/, chỉ để PyG hài lòng

    @property
    def processed_file_names(self):
        return ["processed_data_hariboss.pt"]

    def download(self):
        pass  # không cần

    # def process(self):
    #     # Nếu chưa có cache, bắt buộc phải có dirs để dựng data
    #     if any(d is None for d in [self.pdb_dir, self.fasta_dir,self.label_file_path]):
    #         raise ValueError("Cần pdb_dir, fasta_dir, asa_dir (và emb_dir nếu dùng) cho lần chạy đầu.")

    #     data_list = []

    #     all_edge, all_emb, all_labels, all_ss, allpdb, all_burasa = get_geo_features_test1(
    #         self.pdb_dir, self.fasta_dir,
    #         self.label_file_path, self.topk, self.rna_emb_dir, self.rna_ss_dir, self.asa_path
    #     )

    #     for edge, emb_rna, label, ss, pdbname, burasa in zip(all_edge, all_emb, all_labels, all_ss, allpdb, all_burasa):          
          
    #         edge = torch.tensor(edge, dtype=torch.long)


    #         emb_rna =  torch.tensor(emb_rna, dtype=torch.float)
    #         ss =  torch.tensor(ss, dtype=torch.float)

    #         bur =  torch.tensor(burasa, dtype=torch.float)
          
            

    #         label = torch.tensor(label, dtype=torch.float)
    #         # data = Data(x=feature, edge_index=edge, y=label, rna_embs = emb_rna, ss_emb = ss, onehot_emb = onehot, ed_b1 = edb1, ed_b2 = edb2)
    #         data = Data(
    #             x=emb_rna,
    #             edge_index=edge,
    #             y=label,           
    #             ss_emb=ss,
    #             pdb_name=pdbname,
    #             burasa_emb = bur
    #         )
           
    #         data_list.append(data)
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])

    def process(self):
        # Nếu chưa có cache, bắt buộc phải có dirs để dựng data
        if any(d is None for d in [self.pdb_dir, self.fasta_dir,self.label_file_path]):
            raise ValueError("Cần pdb_dir, fasta_dir, asa_dir (và emb_dir nếu dùng) cho lần chạy đầu.")

        data_list = []

        all_edge, all_emb, all_labels, all_ss, allpdb = get_geo_features_new(
            self.pdb_dir, self.fasta_dir,
            self.label_file_path, self.topk, self.rna_emb_dir, self.rna_ss_dir
        )
        for edge, emb_rna, label, ss, pdbname in zip(all_edge, all_emb, all_labels, all_ss, allpdb):          
          
            edge = torch.tensor(edge, dtype=torch.long)


            emb_rna =  torch.tensor(emb_rna, dtype=torch.float)
            ss =  torch.tensor(ss, dtype=torch.float)
          
            

            label = torch.tensor(label, dtype=torch.float)
            # data = Data(x=feature, edge_index=edge, y=label, rna_embs = emb_rna, ss_emb = ss, onehot_emb = onehot, ed_b1 = edb1, ed_b2 = edb2)
            data = Data(
                x=emb_rna,
                edge_index=edge,
                y=label,           
                ss_emb=ss,
                pdb_name=pdbname
            )
           
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
