# This is a modification of EquiDock: Independent SE(3)-Equivariant Models for End-to-End Rigid Protein Docking (ICLR 2022) with various dropout methods for the course project





@article{ganea2021independent,
  title={Independent SE (3)-Equivariant Models for End-to-End Rigid Protein Docking},
  author={Ganea, Octavian-Eugen and Huang, Xinyuan and Bunne, Charlotte and Bian, Yatao and Barzilay, Regina and Jaakkola, Tommi and Krause, Andreas},
  journal={arXiv preprint arXiv:2111.07786},
  year={2021}
}
```


## Dependencies
Current code works on Linux/Mac OSx only, you need to modify file paths to work on Windows.
```
python==3.9.10
numpy==1.22.1
cuda==10.1
torch==1.10.2
dgl==0.7.0
biopandas==0.2.8
ot==0.7.0
rdkit==2021.09.4
dgllife==0.2.8
joblib==1.1.0
```

#To reproduce the inference result in the project report:
1. prepare DB5.5 dataset:
  Preprocessed DB5.5 dataset for this experiment can be downloaded from . Please move the cache folder into the main path
2. run inference script:
DropEdge: python -m src.inference -dropout 0 -drop_message None -drop_message_rate 0 -drop_connect DropEdge -drop_connect_rate 0.1 -iegmn_n_lays 8 -patience 100 -data db5

DropConnect: python -m src.inference -dropout 0 -drop_message None -drop_message_rate 0 -drop_connect DropConnect -drop_connect_rate 0.1 -iegmn_n_lays 4 -patience 100 -data db5

DropMessage: python -m src.inference -dropout 0 -drop_message DropMessage -drop_message_rate 0.05 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 8 -patience 100 -data db5

DropNode: python -m src.inference -dropout 0 -drop_message DropNode -drop_message_rate 0.25 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 4 -patience 100 -data db5

DropOut: python -m src.inference -dropout 0.05 -drop_message None -drop_message_rate 0 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 4 -patience 100 -data db5

Baseline: python -m src.inference -dropout 0 -drop_message None -drop_message_rate 0 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 8 -patience 100 -data db5



#To reproduce the uncertainty quantification result:

python uq.py




