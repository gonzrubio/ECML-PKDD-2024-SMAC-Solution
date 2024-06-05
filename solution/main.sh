#!/bin/bash

# hyperparameters
MAX_EPOCHS=1000
LOSS_WEIGHTS_LIST=(1 1e-2 1e-4)
# SCHEDULERS=("step" "cosine" "reduce_on_plateau" "cosine_warm" "none")
SCHEDULERS=("reduce_on_plateau")

# for reg_mse_w in "${LOSS_WEIGHTS_LIST[@]}"
# do
#     for class_loss_w in "${LOSS_WEIGHTS_LIST[@]}"
#     do
#         for emb_loss_w in "${LOSS_WEIGHTS_LIST[@]}"
#         do
#             for aux_loss_w in "${LOSS_WEIGHTS_LIST[@]}"
#             do
#                 python main.py trainer.max_epochs=$MAX_EPOCHS \
#                 model.reg_mse_w=$reg_mse_w \
#                 model.class_loss_w=$class_loss_w \
#                 model.emb_loss_w=$emb_loss_w \
#                 model.aux_loss_w=$aux_loss_w
#             done
#         done
#     done
# done

for scheduler in "${SCHEDULERS[@]}"
do
    python main.py trainer.max_epochs=$MAX_EPOCHS \
    model.scheduler=$scheduler
done

