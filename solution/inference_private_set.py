from argparse import ArgumentParser
import pandas as pd
import torch
from model import EarthQuakeModel
from torchgeo.datasets import QuakeSet
from tqdm import tqdm

def main(checkpoint: str):
    # Load the dataset
    dataset = QuakeSet(root="private_set", split="test")
    
    # Load checkpoint
    model = EarthQuakeModel.load_from_checkpoint(checkpoint, map_location="cpu").eval()
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            prediction = model(sample["image"].unsqueeze(0))[0].item()
            if not prediction > 4:
                prediction = 0
            metadata = dataset.data[i]
            predictions.append({
                "key": metadata['key'], 
                "magnitude": prediction, 
                "affected": int(prediction > 1)
            })
    
    # Save predictions to CSV
    pd.DataFrame(predictions).to_csv("submission.csv", index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
        default='checkpoints/20240603-230822/earthquake-detection-epoch=686-val_loss=0.27.ckpt',
        help="Path to the model checkpoint")
    args = parser.parse_args()
    main(args.checkpoint)
