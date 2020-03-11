# python train.py --n_splits 5 --seed 12 --fold 0 --model_type "seresnext50" --loss "ceonehotohem" --lr 2e-1 --num_epoch 100 --batch_size 320 --valid_batch_size 1024 --accumulation_steps 1 --num_workers 14 --early_stopping 20 --apex --load_pretrain --start_epoch 76
 
python train.py --n_splits 5 --seed 12 --fold 1 --model_type "seresnext50" --loss "ceonehotohem" --lr 0.040960 --num_epoch 100 --batch_size 320 --valid_batch_size 1024 --accumulation_steps 1 --num_workers 14 --early_stopping 20 --apex --load_pretrain --start_epoch 63

python train.py --n_splits 5 --seed 12 --fold 2 --model_type "seresnext50" --loss "ceonehotohem" --lr 8e-2 --num_epoch 100 --batch_size 320 --valid_batch_size 1024 --accumulation_steps 1 --num_workers 14 --early_stopping 20 --apex --load_pretrain --start_epoch 3

python train.py --n_splits 5 --seed 12 --fold 3 --model_type "seresnext50" --loss "ceonehotohem" --lr 8e-2 --num_epoch 100 --batch_size 320 --valid_batch_size 1024 --accumulation_steps 1 --num_workers 14 --early_stopping 20 --apex 

python train.py --n_splits 5 --seed 12 --fold 4 --model_type "seresnext50" --loss "ceonehotohem" --lr 8e-2 --num_epoch 100 --batch_size 320 --valid_batch_size 1024 --accumulation_steps 1 --num_workers 14 --early_stopping 20 --apex

# 46 29 26

# python train.py --n_splits 10 --seed 1997 --fold 0 --model_type "efficientnet-b1" --loss "ceonehotohem" --lr 1e-1 --num_epoch 100 --batch_size 320 --valid_batch_size 400 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --apex 

# python train.py --n_splits 10 --seed 1997 --fold 2 --model_type "efficientnet-b1" --loss "ceonehotohem" --lr 1e-1 --num_epoch 100 --batch_size 320 --valid_batch_size 400 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --apex 

# python train.py --n_splits 10 --seed 1997 --fold 4 --model_type "efficientnet-b1" --loss "ceonehotohem" --lr 1e-1 --num_epoch 100 --batch_size 320 --valid_batch_size 400 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --apex 

# python train.py --n_splits 10 --seed 1997 --fold 6 --model_type "efficientnet-b1" --loss "ceonehotohem" --lr 1e-1 --num_epoch 100 --batch_size 320 --valid_batch_size 400 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --apex

# python train.py --n_splits 10 --seed 1997 --fold 8 --model_type "efficientnet-b1" --loss "ceonehotohem" --lr 1e-1 --num_epoch 100 --batch_size 320 --valid_batch_size 400 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --apex


# python train.py --n_splits 10 --seed 2019 --fold 0 --model_type "efficientnet-b3" --loss "ceonehotohem" --lr 2e-1 --num_epoch 100 --batch_size 96 --valid_batch_size 320 --accumulation_steps 2 --num_workers 12 --early_stopping 20 --apex

# python train.py --n_splits 10 --seed 2019 --fold 2 --model_type "efficientnet-b3" --loss "ceonehotohem" --lr 2e-1 --num_epoch 100 --batch_size 96 --valid_batch_size 320 --accumulation_steps 2 --num_workers 12 --early_stopping 20 --apex 

# python train.py --n_splits 10 --seed 2019 --fold 4 --model_type "efficientnet-b3" --loss "ceonehotohem" --lr 2e-1 --num_epoch 100 --batch_size 96 --valid_batch_size 320 --accumulation_steps 2 --num_workers 12 --early_stopping 20 --apex 

# python train.py --n_splits 10 --seed 2019 --fold 6 --model_type "efficientnet-b3" --loss "ceonehotohem" --lr 2e-1 --num_epoch 100 --batch_size 96 --valid_batch_size 320 --accumulation_steps 2 --num_workers 12 --early_stopping 20 --apex

# python train.py --n_splits 10 --seed 2019 --fold 8 --model_type "efficientnet-b3" --loss "ceonehotohem" --lr 2e-1 --num_epoch 100 --batch_size 96 --valid_batch_size 320 --accumulation_steps 2 --num_workers 12 --early_stopping 20 --apex