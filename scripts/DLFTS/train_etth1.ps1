# 检查"./logs"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs"
}
# 检查"./logs/FITS_fa"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs/DLFTS")) {
    New-Item -ItemType Directory -Path "./logs/DLFTS"
}
# 检查"./logs/FITS_fa/etth1_abl"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs/DLFTS/etth1_abl")) {
    New-Item -ItemType Directory -Path "./logs/DLFTS/etth1_abl"
}
# seq_len=700
$model_name = "DLFTS"
$H_order = 6
$seq_len = 720
$pred_len = 720
$m = 1
$seed = 514 #514 1919 810 0 114
$bs = 16#256 #32 64 # 128 256
$features = "M"


& python -u run_longExp_F.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id ETTh1_$seq_len'_'$pred_len `
  --model $model_name `
  --data ETTh1 `
  --features $features `
  --seq_len $seq_len `
  --pred_len $pred_len `
  --enc_in 7 `
  --des 'Exp' `
  --train_mode $m `
  --H_order $H_order `
  --gpu 0 `
  --seed $seed `
  --num_workers 0 `
  --patience 20 `
  --itr 1 --batch_size $bs --learning_rate 0.0005 > logs/$model_name/etth1_abl/$m'_'$model_name'_feature'$features'_'Etth1_$seq_len'_'$pred_len'_H'$H_order'_bs'$bs'_s'$seed.log

echo "Done ${model_name}_Etth1_${seq_len}_${pred_len}_H${H_order}_s${seed}"

