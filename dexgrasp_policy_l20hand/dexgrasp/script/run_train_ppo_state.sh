CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandGrasp \
--algo=ppo \
--seed=7 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test
#--headless
#--model_dir=logs/test_seed6/model_500.pt \
#--test

