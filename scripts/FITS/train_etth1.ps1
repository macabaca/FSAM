# 检查"./logs"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs"
}

# 检查"./logs/FITS_fix/etth1_abl"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs/FITS/etth1_720")) {
    New-Item -ItemType Directory -Path "./logs/FITS/etth1_720"
}
# seq_len=700
$model_name = "FITS"
$H_order = 6
$seq_len = 720
#$pred_len = 720
$m = 2
$seed = 514 #514 1919 810 0 114
$bs = 16#256 #32 64 # 128 256
$features = "M"
$patience = 5

$pred_lens = @("96", "192", "336", "720")
foreach ($pred_len in $pred_lens)
{
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
  --patience $patience `
  --itr 1 --batch_size $bs --learning_rate 0.0005 > logs/FITS/etth1_720/$m'_'$model_name'_feature'$features'_patience_'$patience'_'Etth1_$seq_len'_'$pred_len'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done ${model_name}_Etth1_${seq_len}_${pred_len}_H${H_order}_s${seed}"
}
