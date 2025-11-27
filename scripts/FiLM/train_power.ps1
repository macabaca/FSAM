# 检查"./logs"目录是否存在，如果不存在则创建它
if (!(Test-Path -Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs"
}
$total_files = @("wind_1", "wind_2", "wind_3", "wind_4", "light_1", "light_2", "light_3", "light_4")

foreach ($file in $total_files)
{
  # 检查"./logs/FITS_fix/etth1_abl"目录是否存在，如果不存在则创建它
  if (!(Test-Path -Path "./logs/Film/${file}_abl"))
  {
    New-Item -ItemType Directory -Path "./logs/Film/${file}_abl"
  }
}
# seq_len=700
$model_name = "Film"
$H_order = 6
$seq_len = 32
$pred_len = 32
$m = 1
$seed = 514 #514 1919 810 0 114
$bs = 32#256 #32 64 # 128 256
$features = "M"

$light_files = @("wind_1", "wind_2", "wind_3", "wind_4")
foreach ($file in $light_files)
{
  & python -u run_longExp_F.py `
  --is_training 1 `
  --root_path ../dataset/Power/ `
  --data_path $file'.csv' `
  --model_id $file_$seq_len'_'$pred_len `
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
  --patience 5 `
  --itr 1 --batch_size $bs --learning_rate 0.0005 > logs/Film/${file}_abl/$m'_'$model_name'_feature'$features'_'Etth1_$seq_len'_'$pred_len'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done ${model_name}_${file}_${seq_len}_${pred_len}_H${H_order}_s${seed}"
}

$light_files = @("light_1", "light_2", "light_3", "light_4")
foreach ($file in $light_files)
{
  & python -u run_longExp_F.py `
  --is_training 1 `
  --root_path ../dataset/Power/ `
  --data_path $file'.csv' `
  --model_id $file_$seq_len'_'$pred_len `
  --model $model_name `
  --data custom `
  --features $features `
  --seq_len $seq_len `
  --pred_len $pred_len `
  --enc_in 35 `
  --des 'Exp' `
  --train_mode $m `
  --H_order $H_order `
  --gpu 0 `
  --seed $seed `
  --num_workers 0 `
  --patience 5 `
  --itr 1 --batch_size $bs --learning_rate 0.0005 > logs/Film/${file}_abl/$m'_'$model_name'_feature'$features'_'Etth1_$seq_len'_'$pred_len'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done ${model_name}_${file}_${seq_len}_${pred_len}_H${H_order}_s${seed}"
}
