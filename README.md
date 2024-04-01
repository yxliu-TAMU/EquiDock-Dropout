# This is a modification of EquiDock with various dropout methods for the course project




#To reproduce the inference result in the project report:
1. prepare DB5.5 dataset:
  Preprocessed DB5.5 dataset for this experiment can be downloaded from https://drive.google.com/drive/folders/1mr3J_Qfhzfvbz9ux32suF6Xw3OoVjoal?usp=sharing. Please move the cache folder into the main path
2. run inference script:
DropEdge: python -m src.inference -dropout 0 -drop_message None -drop_message_rate 0 -drop_connect DropEdge -drop_connect_rate 0.1 -iegmn_n_lays 8 -patience 100 -data db5

DropConnect: python -m src.inference -dropout 0 -drop_message None -drop_message_rate 0 -drop_connect DropConnect -drop_connect_rate 0.1 -iegmn_n_lays 4 -patience 100 -data db5

DropMessage: python -m src.inference -dropout 0 -drop_message DropMessage -drop_message_rate 0.05 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 8 -patience 100 -data db5

DropNode: python -m src.inference -dropout 0 -drop_message DropNode -drop_message_rate 0.25 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 4 -patience 100 -data db5

DropOut: python -m src.inference -dropout 0.05 -drop_message None -drop_message_rate 0 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 4 -patience 100 -data db5

Baseline: python -m src.inference -dropout 0 -drop_message None -drop_message_rate 0 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 8 -patience 100 -data db5



#To reproduce the uncertainty quantification result:

python uq.py




