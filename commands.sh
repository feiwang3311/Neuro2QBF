# to generate random QBF instances
python3 randQBF.py --n_quantifiers 2 -a 3 -a 3 -a 8 -a 8 --n_clauses 50 --n_problems 10 --target_dir 3_3_8_8_50_10

# to generate paired random QBF instances (differ by one lit in formula, but differ completely in sat/unsat)
python3 randQBFinc.py --n_quantifiers 2 -a 2 -a 3 -a 2 -a 3 --n_clauses 10 --n_pairs 10 --target_dir 2_3_2_3_10_10 

# to transform dimacs to pickle dump (need to find optimal max_node_per_batch to maximize the efficiency of GPU memory)
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/not.exist --out_dir ./train10/ --max_nodes_per_batch 5000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10

python3 dimacs_to_data.py --dimacs_dir dir/not/exist/ --out_dir ./2_3_2_3_10_10 --max_nodes_per_batch 5000 --n_quantifiers 2 -a 2 -a 3 -a 2 -a 3

# to train
python3 train.py --train_dir ./train10/ --run_id 0 
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 1
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10000 # old GNN (2 LSTM for clause embedding)(Bad: 0.6 at 6000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10001 # flip order of LSTM of clause embedding (OK: 0.1234 at 6000, 0.154 at 3000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10002 # sum message from L and A to update clause embedding (Bad: 0.46 at 6000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10003 # concat message from A and L to update clause embedding (Bad: 0.4865 at 6000, 0.5 at 3000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10004 # concat message and use separate MLP for message CA and message CL (Good: 0.0027 at 6000, 0.0032 at 3000, 0.0532 at 1500)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10010 # new GNN (separate CL and CA embedding)(Good: 0.0484 at 3000)
