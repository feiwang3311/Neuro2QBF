# to generate random QBF instances
python3 randQBF.py --n_quantifiers 2 -a 3 -a 3 -a 8 -a 8 --n_clauses 50 --n_problems 10 --target_dir 3_3_8_8_50_10

# to generate paired random QBF instances (differ by one lit in formula, but differ completely in sat/unsat)
python3 randQBFinc.py --n_quantifiers 2 -a 2 -a 3 -a 2 -a 3 --n_clauses 10 --n_pairs 10 --target_dir 2_3_2_3_10_10 
python3 randQBFinc.py --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --n_clauses 95 --n_pairs 1000 --target_dir train10

# to transform dimacs to pickle dump (need to find optimal max_node_per_batch to maximize the efficiency of GPU memory)
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/ --out_dir ./train10_20/ --max_nodes_per_batch 20000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --max_dimacs 20
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/ --out_dir ./train10_40/ --max_nodes_per_batch 20000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --max_dimacs 40
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/ --out_dir ./train10_80/ --max_nodes_per_batch 40000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --max_dimacs 80
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/ --out_dir ./train10_160/ --max_nodes_per_batch 40000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --max_dimacs 160
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/ --out_dir ./train10_320/ --max_nodes_per_batch 40000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --max_dimacs 320
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/ --out_dir ./train10_1000p_40000/ --max_nodes_per_batch 40000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10
python3 dimacs_to_data.py  --dimacs_dir /homes/wang603/QBF/train10/ --out_dir ./train10_1000p_20000/ --max_nodes_per_batch 20000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10

# to train
python3 train.py --train_dir ./train10/ --run_id 0 
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 1
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10000 # old GNN (2 LSTM for clause embedding)(Bad: 0.6 at 6000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10001 # flip order of LSTM of clause embedding (OK: 0.1234 at 6000, 0.154 at 3000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10002 # sum message from L and A to update clause embedding (Bad: 0.46 at 6000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10003 # concat message from A and L to update clause embedding (Bad: 0.4865 at 6000, 0.5 at 3000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10004 # concat message and use separate MLP for message CA and message CL (Good: 0.0027 at 6000, 0.0032 at 3000, 0.0532 at 1500)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10005 # repeat above at 3000 epochs (Good: 0.0202 at 3000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10010 # new GNN (separate CL and CA embedding)(Good: 0.0484 at 3000)
python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10011 # repeat above at 3000 epochs (Good: 0.147 at 3000)

CUDA_VISIBLE_DEVICES=2 python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10012 # CEGAR like while (while_body2) (Bad: 0.4585 at 3000)
CUDA_VISIBLE_DEVICES=3 python3 train.py --train_dir ./2_3_2_3_10_10/ --run_id 10013 # repeat above (Bad: 0.4533 at 3000) 

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_dir ./train10_20/ --run_id 10 --model_id 0 # train on small set of larger problem (model 0) (Good: 0.0346 at 3000 epochs) 
CUDA_VISIBLE_DEVICES=1 python3 train.py --train_dir ./train10_20/ --run_id 11 --model_id 1 # train on small set of larger problem (model 1) (Good: 0.1518 at 3000 epochs)

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_dir ./train10/ --run_id 12 --model_id 0 # train on 1000 pairs of problems (model 0) 
CUDA_VISIBLE_DEVICES=1 python3 train.py --train_dir ./train10/ --run_id 13 --model_id 1 # train on 1000 pairs of problems (model 1) 
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_dir ./train10_1000p_40000/ --run_id 12 --model_id 0 --restore_id 12 --restore_epoch 5999 # train on 1000 pairs of problems (model 0) (PAUSE for now) 
CUDA_VISIBLE_DEVICES=1 python3 train.py --train_dir ./train10_1000p_20000/ --run_id 13 --model_id 1 --restore_id 13 --restore_epoch 5999 # train on 1000 pairs of problems (model 1) (PAUSE for now)
CUDA_VISIBLE_DEVICES=2 python3 train.py --train_dir ./train10_40/ --run_id 50 --model_id 0 # train on 40 problems (model 0) very good at 1418 epoch
CUDA_VISIBLE_DEVICES=3 python3 train.py --train_dir ./train10_40/ --run_id 51 --model_id 0 --restore_id 10 --restore_epoch 2999 # train on 40 problems (model 0), restoring from the trained NN on 20 problems (not as fast convergence as fresh training, still not so good by 1415 cycle)
CUDA_VISIBLE_DEVICES=2 python3 train.py --train_dir ./train10_80/ --run_id 52 --model_id 0 # train on 80 problems (model 0) not perfect trainign accuracy
CUDA_VISIBLE_DEVICES=3 python3 train.py --train_dir ./train10_80/ --run_id 53 --model_id 0 --restore_id 50 --restore_epoch 1418 # train on 80 problems (model 0), restoring from the trained NN on 40 problems (not perfect training accuracy)
CUDA_VISIBLE_DEVICES=2 python3 train.py --train_dir ./train10_160/ --run_id 54 --model_id 0 # train on 160 problems (model 0)
CUDA_VISIBLE_DEVICES=3 python3 train.py --train_dir ./train10_160/ --run_id 55 --model_id 0 --restore_id 52 --restore_epoch 5028 # train on 160 problems (model 0), restoring from the trained NN on 80 problems
CUDA_VISIBLE_DEVICES=2 python3 train.py --train_dir ./train10_320/ --run_id 56 --model_id 0 # train on 320 problems (model 0)
CUDA_VISIBLE_DEVICES=3 python3 train.py --train_dir ./train10_320/ --run_id 57 --model_id 0 --restore_id 55 --restore_epoch 22892 # train on 320 problems (model 0), restoring from the trained NN on 160 problems
