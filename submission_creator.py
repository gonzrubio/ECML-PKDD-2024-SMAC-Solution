# This script evaluate the format of your submission and add flops

from argparse import ArgumentParser

import pandas as pd


def main(predictions_file: str, flops: int):
    # Load predictions
    predictions = pd.read_csv(predictions_file)

    # Check format
    assert all(
        col in predictions.columns for col in ["key", "magnitude", "affected"]
        ), "Missing columns in predictions file"

    # Check values
    assert predictions["magnitude"].dtype == float, "Magnitude should be a float"
    assert predictions["affected"].dtype == int, "Affected should be an int"
    assert all(
        (0 <= predictions["magnitude"]) & (predictions["magnitude"] <= 10)
        ), "Magnitude should be between 0 and 10"
    assert all(
        (0 <= predictions["affected"]) & (predictions["affected"] <= 1)
        ), "Affected should be 0 or 1"

    # Add flops and save
    predictions["flops"] = flops
    compression_options = dict(method='zip', archive_name='submission.csv')
    predictions.to_csv('submission.zip', compression=compression_options)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictions", type=str, default="solution/submission.csv")
    parser.add_argument("--flops", type=int, default=207594900)
    args = parser.parse_args()
    main(args.predictions, args.flops)
