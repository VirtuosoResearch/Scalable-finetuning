python fast_estimate_eval_approximation_alpaca.py \
    --model_key bigscience/bloomz-560m  \
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.1 --seed 0 --compute_pretrained_outputs --precision "bf16-true" --devices 1  --strategy auto\
    --load_model_dir Alpaca_bigscience-bloomz-560m_meta_train_run_0/epoch_epoch=2\
    --save_name approximation

for seed in 0 1 2 3 4
do
python fast_estimate_eval_approximation_alpaca.py \
    --model_key bigscience/bloomz-560m   --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.1 --seed $seed --devices 1 --strategy auto\
    --load_model_dir Alpaca_bigscience-bloomz-560m_meta_train_run_0/epoch_epoch=2\
    --save_name approximation

python fast_estimate_eval_approximation_alpaca.py \
    --model_key bigscience/bloomz-560m   --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.08 --seed $seed --devices 1 --strategy auto\
    --load_model_dir Alpaca_bigscience-bloomz-560m_meta_train_run_0/epoch_epoch=2\
    --save_name approximation

python fast_estimate_eval_approximation_alpaca.py \
    --model_key bigscience/bloomz-560m   --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.06 --seed $seed --devices 1 --strategy auto\
    --load_model_dir Alpaca_bigscience-bloomz-560m_meta_train_run_0/epoch_epoch=2\
    --save_name approximation

python fast_estimate_eval_approximation_alpaca.py \
    --model_key bigscience/bloomz-560m   --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.04 --seed $seed --devices 1 --strategy auto\
    --load_model_dir Alpaca_bigscience-bloomz-560m_meta_train_run_0/epoch_epoch=2\
    --save_name approximation

python fast_estimate_eval_approximation_alpaca.py \
    --model_key bigscience/bloomz-560m   --precision "bf16-true"\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.02 --seed $seed --devices 1 --strategy auto\
    --load_model_dir Alpaca_bigscience-bloomz-560m_meta_train_run_0/epoch_epoch=2\
    --save_name approximation
done
