import os, urllib.request

import requests

import os
import requests
from Bio import PDB
from io import StringIO

import os
import requests
from Bio import PDB
from io import StringIO

def download_pdb_all_chains(pdb_id, out_dir):
    pdb_id = pdb_id.upper()
    os.makedirs(out_dir, exist_ok=True)

    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        response = requests.get(pdb_url)
        if response.status_code == 404:
            print(f"⚠️ PDB {pdb_id} not found (404).")
            return None
    except requests.RequestException as e:
        print(f"⚠️ Error downloading {pdb_id}: {e}")
        return None

    pdb_content = response.text

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(pdb_content))

    io = PDB.PDBIO()
    io.set_structure(structure)

    pdb_output = os.path.join(out_dir, f"{pdb_id}.pdb")
    io.save(pdb_output)   # ⬅️ không select → giữ tất cả chain

    print(f"Saved ALL chains of {pdb_id} to {pdb_output}")
    return pdb_output

PDB_ID = "5V3F"
out_dir = "interpre/in/pdb"

download_pdb_all_chains(PDB_ID, out_dir)
print("Downloaded")
