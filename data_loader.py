from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, crop_size=178, image_size=128, 
               batch_size=16,  mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ColorJitter(brightness=(0.8, 1.4), contrast=0.2))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader



class neutral_dataset(data.Dataset):
    def __init__(self, sample_label_dir, transform):
        super().__init__()
        self.sample_label_dir = sample_label_dir
        self.transform = transform
        self.dataset = ImageFolder(self.sample_label_dir, transform)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x,y
    
def get_sample(config):
    transform = []
    transform.append(T.CenterCrop(config.crop_size))
    transform.append(T.Resize(config.image_size))
    transform.append(T.ColorJitter(brightness=(0.8, 1.4), contrast=0.2))
    transform.append(T.GaussianBlur(kernel_size=3, sigma=(0.01, 1)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = neutral_dataset(config.sample_label_dir, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers)
    return data_loader

def get_fid_sample(config):
    transform = []
    transform.append(T.CenterCrop(config.crop_size))
    transform.append(T.Resize(config.image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    images_tensor = torch.zeros((config.fid_sample_size, 3, config.image_size, config.image_size))
    image_files = os.listdir(config.fid_sample_dir)[:config.fid_sample_size]
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_tensor = transform(img)
            images_tensor[i] = img_tensor
    return images_tensor