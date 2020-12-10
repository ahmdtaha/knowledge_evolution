import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs_path, lbls,is_training):
        self.imgs_path = imgs_path
        self.lbls = lbls
        #self.idx = list(range(0,len(lbls)))
        if is_training:
            self.transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            img = Image.open(self.imgs_path[idx])
            if len(img.mode) != 3 or len(img.getbands())!=3:
                # print(len(img.mode),len(img.getbands()))
                # print(self.imgs_path[idx])
                img = img.convert('RGB')

            img = self.transform(img)
            # sample = {'image': self.transform(img), 'label': self.lbls[idx],'index': idx}
        else:
            img = Image.open(self.imgs_path[idx])
            # sample = {'image': Image.open(self.imgs_path[idx]), 'label': self.lbls[idx],'index': idx}

        return img,self.lbls[idx]
