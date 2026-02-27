import torch
import numpy as np
from torch_geometric.loader import DataLoader
from feature_extraction.datasetnew import RNAGraphDatasetNew
from model.RNABP import HybridRNABindingSiteModel
from Bio.PDB import PDBParser, PDBIO

def read_id_list(path):
    """Đọc mỗi dòng trong file txt thành một ID (bỏ dòng trống / khoảng trắng)."""
    ids = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids

TARGET = "8hb8"
PDB_PATH = r"interpre/in/pdb/8hb8.pdb"   # sửa thành đường dẫn file PDB của bạn


device = "cpu"

# TE18NEW: processed/TE18NEW
print("Loading test dataset...")
test_ids  = set(read_id_list("data/HARIBOSS/SET3/test.txt"))
print(f"Số ID trong test.txt:  {len(test_ids)}")
print("Loading test dataset...")
test_dataset  = []

full_dataset = RNAGraphDatasetNew(
    root='processed/HARI_FINAL'
)
for data in full_dataset:
    pdb_name = getattr(data, "pdb_name", None)  # đổi nếu bạn đặt tên khác
    if pdb_name is None:
        raise ValueError("Một sample trong full_dataset không có thuộc tính 'pdb_name'")

    # Nếu id nằm trong list nào thì cho vào dataset đó
    if pdb_name in test_ids:
        test_dataset.append(data)


    # 4) In thông tin
    print(f"Test dataset size: {len(test_dataset)}")



# tìm sample đúng pdb_name
idx = None
for i in range(len(test_dataset)):
    if test_dataset[i].pdb_name.lower() in TARGET.lower():
        idx = i
        break
assert idx is not None, f"Không tìm thấy {TARGET} trong dataset"

data = test_dataset[idx].to(device)

model = HybridRNABindingSiteModel(
    rna_dim=test_dataset[0].x.shape[1],
    ss_dim=test_dataset[0].ss_emb.shape[1],
    hidden=86,
    dropout=0.3
).to(device)

ckpt = torch.load('HARI_SET3_best_model.pt', map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(data.x, data.ss_emb, data.edge_index, data.batch)).cpu().numpy()

print("✓ N nodes:", len(probs))

parser = PDBParser(QUIET=True)
structure = parser.get_structure("RNA", PDB_PATH)

# resname nucleotide chuẩn hay gặp
NUC_RESNAMES = {"A","C","G","U","I", "DA","DC","DG","DT","DU"}

# atom đường đặc trưng cho nucleotide (giúp nhận diện nucleotide biến đổi)
SUGAR_ATOMS = {"C1'", "C2'", "C3'", "C4'", "O4'", "O3'", "O5'", "C5'"}

# loại thẳng các ligand hay gây nhiễu (có thể bổ sung nếu gặp thêm)
EXCLUDE_RESN = {"B12", "H_B12"}

def is_nucleotide_like(res):
    resn = res.get_resname().strip()
    if resn in EXCLUDE_RESN:
        return False
    if resn in {"HOH", "WAT"}:
        return False
    if resn in NUC_RESNAMES:
        return True
    # nucleotide biến đổi: thường vẫn có atom đường ribose
    if any(a in res for a in SUGAR_ATOMS):
        return True
    return False

# collect residues trong chain mục tiêu
residues = []
for m in structure:
    for chain in m: # Duyệt qua mọi chain có trong model
        for res in chain:
            if is_nucleotide_like(res):
                residues.append(res)
    break # Thường chỉ lấy model đầu tiên (index 0)

print("✓ N residues in PDB chain (nucleotide-like):", len(residues))
print("✓ N nodes from model:", len(probs))

# ====== WRITE PDB WITH B-FACTOR ======
def write_pdb_with_bfactor(structure, residues, node_scores, out_pdb):
    if len(node_scores) != len(residues):
        raise ValueError(
            f"Mismatch: N nodes={len(node_scores)} != N residues={len(residues)} "
            f"→ mapping theo thứ tự sẽ sai. Hãy kiểm tra lọc residue / chain."
        )

    for i, res in enumerate(residues):
        score = float(node_scores[i])
        for atom in res.get_atoms():
            atom.set_bfactor(score)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)

OUT_PDB = f"interpre/out/{TARGET}_pred_bfac.pdb"
write_pdb_with_bfactor(structure, residues, probs, OUT_PDB)
print("✅ Saved PDB for PyMOL:", OUT_PDB)

# debug list
print("---- residues included ----")
for i, r in enumerate(residues):
    hetflag, resseq, icode = r.id
    print(i, r.get_resname().strip(), hetflag, resseq, icode,
          "hasP" if "P" in r else "-",
          "hasC4'" if "C4'" in r else "-")


#  CHECK ATOMS
LABEL_PATH = rf"data/HARIBOSS/FINAL/LABELS/{TARGET}.txt"  # sửa lại đúng folder của bạn
# ====== READ GT LABELS FROM TXT ======
labs = []
with open(LABEL_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        labs.append(int(float(parts[1])))

print("✓ N labels from txt:", len(labs), " unique:", set(labs))
# ====== BUILD NUCLEOTIDE LIST BY C4' ORDER (1 nucleotide ~ 1 C4') ======
model0 = next(structure.get_models())
residues_by_c4 = []
seen = set()

for chain in model0: # Duyệt tất cả chain
    for res in chain:
        if "C4'" in res: 
            # Tạo unique ID bao gồm cả Chain ID để tránh trùng giữa các chain
            unique_rid = (chain.id, res.id) 
            if unique_rid not in seen:
                residues_by_c4.append(res)
                seen.add(unique_rid)

print(f"✓ N nucleotides by C4' in ALL chains: {len(residues_by_c4)}")

# Kiểm tra khớp với file label txt
assert len(labs) == len(residues_by_c4), (
    f"Label length ({len(labs)}) != N nucleotides by C4' ({len(residues_by_c4)})"
)
gt_res_cnt = 0
gt_atom_cnt = 0

for i, res in enumerate(residues_by_c4):
    if labs[i] == 1:
        gt_res_cnt += 1
        gt_atom_cnt += sum(1 for _ in res.get_atoms())

print("✅ GT residues (label=1):", gt_res_cnt)
print("✅ GT atoms (sum atoms in label=1 residues):", gt_atom_cnt)
