""" File to preprocess Digit datasets
- Download the datasets
- preprocess and flatten the images
- Store the preprocessed data in a dictionary and save it in a pickle file.
- The pickle file is stored in the datasets folder.
"""

import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
import torch
import torchvision
from torchvision.datasets import MNIST, SVHN, USPS
from transformers import ViTImageProcessor, ViTModel


if __name__ == "__main__":
    DATASETS = ['MNIST', 'SVHN', 'USPS']
    RANDOM_STATE = 27
    BATCH_SIZE = 1024
    N_COMPONENTS = 100
    N_JOBS = os.cpu_count()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CLIP model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k').to(DEVICE)
    model.eval()

    preprocessed_data = dict()
    for dataset_name in DATASETS:
        print(f"Preprocessing {dataset_name} dataset")

        dataset_name = dataset_name.lower()

        if dataset_name == 'mnist':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])
            dataset = MNIST(
                root='./data/MNIST',
                download=True,
                train=True,
                transform=transform
            )
        elif dataset_name == 'svhn':
            dataset = SVHN(
                root='./data/SVHN',
                download=True,
                split='train',
                transform=torchvision.transforms.ToTensor()
            )
        elif dataset_name == 'usps':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])
            dataset = USPS(
                root='./data/USPS',
                download=True,
                train=True,
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        # Generate embeddings using CLIP
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=N_JOBS
        )
        embeddings = list()
        with torch.no_grad():
            for images, _ in dataloader:
                inputs = processor(images=[image for image in images], return_tensors="pt")
                inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
                outputs = model(**inputs).last_hidden_state[..., 0, :]  # cls token
                embeddings.append(outputs.cpu().numpy().astype(np.float64))

                del inputs, outputs

        embeddings = np.concatenate(embeddings, axis=0)

        # Save the preprocessed data
        preprocessed_data[dataset_name] = {
            'X': embeddings,
            'y': np.concatenate([y for _, y in dataloader])
        }

    # Apply PCA to reduce the dimensionality of the embeddings
    print("Applying PCA...")
    full_X = np.concatenate(
        [
            preprocessed_data[dataset_name.lower()]['X']
            for dataset_name in DATASETS
        ],
        axis=0
    )
    pca = PCA(
        n_components=N_COMPONENTS,
        whiten=True,
        random_state=RANDOM_STATE
    )
    pca.fit(full_X)
    for dataset_name in DATASETS:
        preprocessed_data[dataset_name.lower()]['X'] = pca.transform(
            preprocessed_data[dataset_name.lower()]['X']
        )

    # Save the preprocessed data in a pickle file
    print("Saving the preprocessed data...")
    path = Path('data')
    path.mkdir(exist_ok=True)
    with open(path / 'digit_preprocessed.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
