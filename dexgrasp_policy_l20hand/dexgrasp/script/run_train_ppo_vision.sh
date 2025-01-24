#CUDA_VISIBLE_DEVICES=0 \
#python train.py \
#--task=ShadowHandRandomLoadVision \
#--algo=ppo \  #任务类型
#--seed=2 \    #随机种子
#--rl_device=cuda:0 \
#--sim_device=cuda:0 \
#--logdir=logs/test \
#--headless \
#--vision \
#--backbone_type=pn
##pn/transpn

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandRandomLoadVision \
--algo=ppo \
--seed=1 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--model_dir=example_model/model.pt \
--backbone_type=pn \
--headless \
--vision
