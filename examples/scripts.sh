
cd path/to/GFSLT-VLP
# cd examples/GFSLT-VLP

# check SignCL pretrained
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 --use_env train_vlp_SignCL.py --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/0626_csl_VLP_SignCL --training-refurbish True --noise-rate 0.15 --noise-type omit_last --random-shuffle False --decoder-type LLMD --config ./configs/config_gloss_free_csl.yaml


# check best approach on CSL-Daily 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1036 --use_env train_csl_SignCL.py --batch-size 2 --epochs 200 --opt sgd --lr 0.0065 --output_dir out/0630_GF_SignCL \
--config ./configs/config_gloss_free_csl.yaml --finetune out/0626_csl_VLP_SignCL/best_checkpoint.pth --decoder-type LLMD

# eval
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1036 --use_env train_csl_SignCL.py --batch-size 2 --epochs 0 --opt sgd --lr 0.0065 --output_dir out/0630_GF_SignCL \
--config ./configs/config_gloss_free_csl_char.yaml --resume out/0630_GF_SignCL/best_checkpoint.pth --decoder-type LLMD --eval


