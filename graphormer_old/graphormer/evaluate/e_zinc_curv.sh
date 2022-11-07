python evaluate.py \
    --user-dir ../../graphormer \
    --num-workers 8 \
    --ddp-backend=legacy_ddp \
    --dataset-name zinc \
    --dataset-source pyg \
    --task graph_prediction \
    --arch graphormer_slim \
    --num-classes 1 \
    --batch-size 64 \
    --save-dir ../../../examples/property_prediction/ckpts/zinc_curvnew/ckpt_e \
    --split test \
    --metric mae \
    --seed 1 \
    # --pre-layernorm