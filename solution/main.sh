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

for REGRESSION_LOSS_COEFFICIENTS in "${REGRESSION_LOSS_COEFFICIENTS_LIST[@]}"
do
    python main.py trainer.max_epochs=$MAX_EPOCHS model.regression_loss="$REGRESSION_LOSS" model.regression_loss_coefficients="$REGRESSION_LOSS_COEFFICIENTS"
done
