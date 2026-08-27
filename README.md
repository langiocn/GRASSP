# GRASSP: RNA Language Model–Enhanced Graph Attention with Adaptive Gating for RNA–Small Molecule Binding Site Prediction

## 1. Setup
```bash
git clone https://github.com/langiocn/GRASSP.git
cd GRASSP
conda env create -f environment.yml
conda activate GRASSP
````

## 2. Run Test
Each dataset has a corresponding test script in the `test/` directory.
```bash
python test/testTE18.py      # TE18 dataset
````
## 3. Citation

If you find this work useful in your research, please cite our paper as follows:
```bibtex
Nguyen, Thi Lan, and Nguyen Quoc Khanh Le. "GRASSP: RNA Language Model–Enhanced Graph Attention with Adaptive Gating for RNA–Small Molecule Binding Site Prediction." Bioinformatics (2026): btag638.
