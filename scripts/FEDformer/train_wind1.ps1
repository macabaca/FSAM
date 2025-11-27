# 检查"./logs"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs"
}

# 检查"./logs/FITS_fix/etth1_abl"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs/FEDformer/wind1_abl")) {
    New-Item -ItemType Directory -Path "./logs/FEDformer/wind1_abl"
}
# seq_len=700
$model_name = "FEDformer"
$H_order = 6
$seq_len = 32
$pred_len = 32
$m = 1
$seed = 514 #514 1919 810 0 114
$bs = 1#256 #32 64 # 128 256
$features = "M"


& python -u run_longExp_F.py `
  --is_training 1 `
  --root_path ../dataset/Power/ `
  --data_path wind_1.csv `
  --model_id wind1_$seq_len'_'$pred_len `
  --model $model_name `
  --data custom `
  --features $features `
  --seq_len $seq_len `
  --pred_len $pred_len `
  --enc_in 96 `
  --des 'Exp' `
  --train_mode $m `
  --H_order $H_order `
  --gpu 0 `
  --seed $seed `
  --num_workers 0 `
  --patience 30 `
  --itr 1 --batch_size $bs --learning_rate 0.0005 > logs/FEDformer/wind1_abl/$m'_'$model_name'_feature'$features'_'Etth1_$seq_len'_'$pred_len'_H'$H_order'_bs'$bs'_s'$seed.log

echo "Done ${model_name}_wind1_${seq_len}_${pred_len}_H${H_order}_s${seed}"

