"""Exploratory Data Analysis for QuakeSet

VV and VH polarization mean and std (of training set) for standardizing input:
    - mean [0.1048, 0.0189] for hh and hv (or [0.1048, 0.0189, 0.1048, 0.0189])
    - std [0.8203, 0.0758] " " " " (or [0.8203, 0.0758, 0.8203, 0.0758])

"""

import os
import random
import matplotlib.pyplot as plt
import torch
from torchgeo.datasets import QuakeSet


#  Load the datasets
train_dataset = QuakeSet(root=os.path.join("..", "data"), split="train", download=False)
val_dataset = QuakeSet(root=os.path.join("..", "data"), split="val", download=False)
test_dataset = QuakeSet(root=os.path.join("..", "data"), split="test", download=False)

# Plot some examples for all splits
def plot_random_examples(dataset, num_examples):
    for i in random.sample(range(len(dataset)), num_examples):
        sample = dataset[i]
        # sample["image"], sample["label"], sample["magnitude"]
        # metadata = dataset.data[i]
        # metadata['key'], metadata['patch'], metadata['images']
        # metadata['label'], metadata['magnitude']
        dataset.plot(sample, show_titles=True)

num_examples = 2
plot_random_examples(train_dataset, num_examples)
plot_random_examples(val_dataset, num_examples)
plot_random_examples(test_dataset, num_examples)

# Combined distribution of earthquake magnitudes
nans = 0
negative = 0
magnitudes = []
for i, sample in enumerate(train_dataset+val_dataset+test_dataset):
    if  torch.isnan(sample["magnitude"]):
        nans += 1
    elif sample["magnitude"] > 0:
        magnitudes.append(sample["magnitude"].item())
    elif sample["magnitude"] == 0:
        negative += 1
    else:
        print('weird magnitude')

plt.figure()
plt.hist(magnitudes, bins=14, color='skyblue', edgecolor='black')
plt.title(f'positive: {len(magnitudes)}, negative: {negative}, nan: {nans}')
plt.xlabel('magnitude (mb)')
plt.ylabel('Count')
plt.show()

# min, max and mean backscatter for all splits (over all pols and time samples) 
def backscatter_stats(dataset):
    min_backscatter_values = []
    max_backscatter_values = []
    mean_backscatter_values = 0

    for sample in dataset:
        image = sample["image"]
        if torch.isnan(image).any():
            print('nan(s)')
        elif not torch.is_floating_point(image):
                print('not float')
        else:
            min_backscatter_values.append(image.min().item())
            max_backscatter_values.append(image.max().item())
            mean_backscatter_values += image.mean().item()

    min_b = min(min_backscatter_values)
    max_b = max(max_backscatter_values)
    mean_b = mean_backscatter_values / len(dataset)

    return min_b, max_b, mean_b

train_min, train_max, train_mean = backscatter_stats(train_dataset)
val_min, val_max, val_mean = backscatter_stats(val_dataset)
test_min, test_max, test_mean = backscatter_stats(test_dataset)

print("Train dataset:")
print("Min backscatter:", train_min)
print("Max backscatter:", train_max)
print("Mean backscatter:", train_mean)

print("\nValidation dataset:")
print("Min backscatter:", val_min)
print("Max backscatter:", val_max)
print("Mean backscatter:", val_mean)

print("\nTest dataset:")
print("Min backscatter:", test_min)
print("Max backscatter:", test_max)
print("Mean backscatter:", test_mean)

# calculate pol-wise mean and std for each split
def mean_std(dataset):
    # var[X] = E[X**2] - E[X]**2
    mean = [0, 0]        # VV
    square_sum = [0, 0]  # VH

    for sample in dataset:
        image = sample["image"]  # [4 x 512 x 512]
        mean[0] += image[[0, 2], :, :].sum()
        mean[1] += image[[1, 3], :, :].sum()
        square_sum[0] += (image[[0, 2], :, :] ** 2).sum()
        square_sum[1] += (image[[1, 3], :, :] ** 2).sum()

    numel = len(dataset) * 2 * dataset[0]['image'].shape[-1] ** 2
    mean = torch.tensor(mean) / numel
    std = (torch.tensor(square_sum) / numel - mean**2) ** 0.5

    return mean, std

train_mean, train_std = mean_std(train_dataset)
val_mean, val_std = mean_std(val_dataset)
test_mean, test_std = mean_std(test_dataset)

print("\nMean and standard deviation for train dataset:")
print("Mean:", train_mean)
print("Standard Deviation:", train_std)
print("\nMean and standard deviation for validation dataset:")
print("Mean:", val_mean)
print("Standard Deviation:", val_std)
print("\nMean and standard deviation for test dataset:")
print("Mean:", test_mean)
print("Standard Deviation:", test_std)
