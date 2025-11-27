# export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/FSAM" ]; then
    mkdir ./logs/FSAM
fi

if [ ! -d "./logs/FSAM/weather_test" ]; then
    mkdir ./logs/FSAM/weather_test
fi
model_name=FSAM
base_T=30
data_root=./dataset/Weather
mask_file=mask/stats/prob/weather_720_lag1.pt
patch_len=48
seq_len=720
dropout=0.6
d_model=512
n_heads=1
d_ff=1024
e_layers=2
mask_id=2
m=1
seed=114 #514 1919 810 0
bs=16 #256 #32 64 # 128 256

for d_ff in 512
do
for n_heads in 1
do
for e_layers in 2
do
for threshold in 0.5
do
for stride in 1
do
  pred_len=96
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path weather.csv \
    --model_id weather_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --dropout $dropout \
    --threshold $threshold \
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
    --itr 1 --batch_size $bs --learning_rate 0.00005 | tee logs/FSAM/weather_test/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'weather_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*

  pred_len=192
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path weather.csv \
    --model_id weather_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --dropout $dropout \
    --threshold $threshold \
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
    --itr 1 --batch_size $bs --learning_rate 0.00004 | tee logs/FSAM/weather_test/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'weather_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*


  pred_len=336
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path weather.csv \
    --model_id weather_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --dropout $dropout \
    --threshold $threshold \
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
    --itr 1 --batch_size $bs --learning_rate 0.00003 | tee logs/FSAM/weather_test/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'weather_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*


  pred_len=720
  python -u run_longExp_F.py \
    --is_training 1 \
    --root_path $data_root \
    --data_path weather.csv \
    --model_id weather_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --train_mode $m \
    --patch_len $patch_len \
    --stride $stride \
    --revin 1 \
    --dropout $dropout \
    --threshold $threshold \
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
    --itr 1 --batch_size $bs --learning_rate 0.000015 | tee logs/FSAM/weather_test/$seq_len'_'$pred_len'_threshold'$threshold'_patch_len'$patch_len'_dropout'$dropout'_d_model'$d_model'_n_heads'$n_heads'_d_ff'$d_ff'_e_layers'$e_layers'_stride'$stride'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'weather_$seq_len'_'$pred_len'_H'$H_order'_s'$seed"
  rm -rf results/*
done
done
done
done
done
