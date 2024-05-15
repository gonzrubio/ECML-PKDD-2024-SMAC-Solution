#!/bin/bash


# hyperparameters
MAX_EPOCHS=1000
REGRESSION_LOSS="MAE+MSE"
REGRESSION_LOSS_COEFFICIENTS_LIST=(
    "[1, 1e-2]"
    "[1, 1e-1]"
    "[1, 0.5]"
    "[1, 1]"
)

CLASSIFICATION_LOSS_COEFFICIENT_LIST=(
    "1e-3"
    "1e-2"
    "1e-1"
    "1"
)

for CLASSIFICATION_LOSS_COEFFICIENT in "${CLASSIFICATION_LOSS_COEFFICIENT_LIST[@]}"
do
    python main.py trainer.max_epochs=$MAX_EPOCHS model.regression_loss="$REGRESSION_LOSS" model.classification_loss_coefficient="$CLASSIFICATION_LOSS_COEFFICIENT"
done
