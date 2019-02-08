import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd

from PIL import Image
from os import listdir
from os.path import isfile, join, splitext


def get_split(split_name, dataset_dir, image_size):
    if split_name == 'train':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
    elif split_name == 'val':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]))
    else:
        raise ValueError("Split name '%s' unknown" % split_name)
    return dataset


class SnapshotSerengetiDataset(data.Dataset):

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

    def __init__(self, root, transform=None):
        """ Dataloader for the Snapshot Serengeti dataset.

        Directory structure:

        root/
            images/
            labels.csv
        """

        labels = pd.read_csv(join(root, 'labels.csv'))

        species = sorted(labels.Species.unique().tolist())
        species_to_idx = dict(species, range(len(species)))

        count = sorted(labels.Count.unique().tolist())
        count_to_idx = dict(count, range(len(count)))

        img_dir = join(root, 'images')
        images = []
        for file in listdir(img_dir):
            path = join(img_dir, file)
            img_id, ext = splitext(file)
            if isfile(path) and ext.lower() in IMG_EXTENSIONS:
                rows = labels.loc[labels.CaptureEventID == img_id]
                if len(rows) == 1:
                    r = rows.values.tolist()[0]
                    item = (path, species_to_idx[r[1]], count_to_idx[r[2]])
                    images.append(item)

        self.root = root
        self.loader = pil_loader
        self.transform = transform

        self.classes = (species, count)
        self.classes_to_idx = (species_to_idx, count_to_idx)

        self.images = images
        self.targets = [(i[1], i[2]) for i in images]


    def __getitem__(self, index):
        path, species, count = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, species, count


    def __len__(self):
        return len(self.images)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace(
            '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def is_valid_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
