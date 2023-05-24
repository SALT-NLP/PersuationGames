model=bert
dataset=Youtube

for bs in 16 8
do
  for lr in 1e-5 3e-5 5e-5
  do
    for context_size in 0
    do
      for seed in 13 42 87
      do
        CUDA_VISIBLE_DEVICES=0 python3 baselines/main.py \
        --dataset ${dataset} \
        --model_type ${model} \
        --model_name bert-base-uncased \
        --batch_size ${bs} \
        --learning_rate ${lr} \
        --context_size ${context_size} \
        --seed ${seed} \
        --output_dir out/${dataset}/${model}/${bs}_${lr}/${seed} \
#        --avalon
      done
    done
  done
done