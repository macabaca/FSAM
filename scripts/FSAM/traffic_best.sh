export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/FSAM" ]; then
    mkdir ./logs/FSAM
fi

if [ ! -d "./logs/FSAM/traffic_test_720" ]; then
    mkdir ./logs/FSAM/traffic_test_720
fi
# seq_len=700
model_name=FSAM
base_T=30
data_root=./dataset/Traffic
mask_file=mask/stats/prob/traffic_720_lag1.pt
patch_len=48
seq_len=720
dropout=0
threshold=0.15
d_model=512
n_heads=1
d_ff=512
e_layers=2
mask_id=2
m=1
seed=114 #514 1919 810 0
# bs=16 #256 #32 64 # 128 256

for stride in 6
do
for threshold in 0.15
do
for bs in 16
do
for lr in 0.0013
do
for dropout in 0
do

  pred_len=96
  threshold=0.1
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path traffic.csv \
    --model_id traffic_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --threshold $threshold \
    --dropout $dropout \
    --d_model $d_model \
    --n_heads $n_heads \
    --mask_file $mask_file \
    --d_ff $d_ff \
    --e_layers $e_layers \
    --mask_id $mask_id \
    --upsample_complex 1 \
    --if_complex 0 \
    --gpu 0 \
    --seed $seed \
    --patience 9 \
    --itr 1 --batch_size $bs --learning_rate 0.001 | tee logs/FSAM/traffic_test_1/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'traffic_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*

  pred_len=192
  threshold=0.1
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path traffic.csv \
    --model_id traffic_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --threshold $threshold \
    --dropout $dropout \
    --d_model $d_model \
    --n_heads $n_heads \
    --mask_file $mask_file \
    --d_ff $d_ff \
    --e_layers $e_layers \
    --mask_id $mask_id \
    --upsample_complex 1 \
    --if_complex 0 \
    --gpu 0 \
    --seed $seed \
    --patience 9 \
    --itr 1 --batch_size $bs --learning_rate 0.001 | tee logs/FSAM/traffic_test_1/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'traffic_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*
  
  pred_len=336
  threshold=0.15
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path traffic.csv \
    --model_id traffic_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --threshold $threshold \
    --dropout $dropout \
    --d_model $d_model \
    --n_heads $n_heads \
    --mask_file $mask_file \
    --d_ff $d_ff \
    --e_layers $e_layers \
    --mask_id $mask_id \
    --upsample_complex 1 \
    --if_complex 0 \
    --gpu 0 \
    --seed $seed \
    --patience 9 \
    --itr 1 --batch_size $bs --learning_rate 0.0007 | tee logs/FSAM/traffic_test_1/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'traffic_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*

  
  pred_len=720
  threshold=0.1
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path traffic.csv \
    --model_id traffic_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --threshold $threshold \
    --dropout $dropout \
    --d_model $d_model \
    --n_heads $n_heads \
    --mask_file $mask_file \
    --d_ff $d_ff \
    --e_layers $e_layers \
    --mask_id $mask_id \
    --upsample_complex 1 \
    --if_complex 0 \
    --gpu 0 \
    --seed $seed \
    --patience 9 \
    --itr 1 --batch_size $bs --learning_rate 0.002 | tee logs/FSAM/traffic_test_720/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'traffic_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*
  

done
done
done
done
done
