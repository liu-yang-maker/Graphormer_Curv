python /mnt/sfs_turbo/liuyang/Lai/Graphormer/graphormer_old/graphormer/evaluate/evaluate.py \
    --user-dir ../../graphormer_old/graphormer \
    --num-workers 8 \
    --ddp-backend=legacy_ddp \
    --dataset-name ogbg-molhiv \
    --dataset-source ogb \
    --task graph_prediction_with_flag \
    --arch graphormer_base \
    --num-classes 1 \
    --encoder-layers 18 \
    --batch-size 64 \
    --save-dir /mnt/sfs_turbo/liuyang/Lai/Graphormer/examples/property_prediction/ckpts/hiv/eva \
    --split test \
    --metric auc \
    --seed 1 \
    # --pre-layernorm