# python train.py --n_splits 5 --seed 12 --fold 0 --model_type "seresnext50" --loss "ceonehotohem" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20
# python train.py --n_splits 5 --seed 12 --fold 0 --model_type "seresnext50" --loss "ceonehot" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --load_pretrain


# python train.py --n_splits 5 --seed 12 --fold 1 --model_type "seresnext50" --loss "ceonehotohem" --lr 4e-3 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --load_pretrain --start_epoch 50
# python train.py --n_splits 5 --seed 12 --fold 1 --model_type "seresnext50" --loss "ceonehot" --lr 3e-1  --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --load_pretrain


python train.py --n_splits 5 --seed 12 --fold 2 --model_type "seresnext50" --loss "ceonehotohem" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20
# python train.py --n_splits 5 --seed 12 --fold 2 --model_type "seresnext50" --loss "ceonehot" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --load_pretrain

python train.py --n_splits 5 --seed 12 --fold 3 --model_type "seresnext50" --loss "ceonehotohem" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20
# python train.py --n_splits 5 --seed 12 --fold 3 --model_type "seresnext50" --loss "ceonehot" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --load_pretrain


python train.py --n_splits 5 --seed 12 --fold 12 --model_type "seresnext50" --loss "ceonehotohem" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20
# python train.py --n_splits 5 --seed 12 --fold 12 --model_type "seresnext50" --loss "ceonehot" --lr 3e-1 --num_epoch 100 --batch_size 384 --valid_batch_size 800 --accumulation_steps 1 --num_workers 12 --early_stopping 20 --load_pretrain