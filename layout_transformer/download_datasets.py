import os
from torchvision import datasets, transforms
import idx2numpy

# Define the directory where the dataset should be located
data_dir = "/home/hepe00001/Desktop/neuro_explicit/generative_diffusion/datasets/MNIST/raw"

# Check if the dataset has already been downloaded
if not os.path.exists(data_dir):
    print("Dataset not found. Downloading...")
    
    # Define transformations (convert to tensor in this case)
    transform = transforms.ToTensor()
    
    # Download and load the training and test datasets
    train_data = datasets.MNIST(root=data_dir, download=True, train=True, transform=transform)
    test_data = datasets.MNIST(root=data_dir, download=True, train=False, transform=transform)
    
    print("Download complete.")
else:
    print("Dataset already exists.")
    
    train_images_file = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images_file = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

    # Load the images and labels
    train_images = idx2numpy.convert_from_file(train_images_file)
    train_labels = idx2numpy.convert_from_file(train_labels_file)
    test_images = idx2numpy.convert_from_file(test_images_file)
    test_labels = idx2numpy.convert_from_file(test_labels_file)

    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

# You can now use the dataset as train_data and test_data
print(f"Training data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")
