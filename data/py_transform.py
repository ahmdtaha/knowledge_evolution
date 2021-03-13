from PIL import Image, ImageOps
import torchvision.transforms as T

class Transform_single():
    def __init__(self, cfg ,image_size, train, mean_std):
        if train == True:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*mean_std)
            ])
        else:
                self.transform = T.Compose([
                    T.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256
                    T.Resize(int(image_size*(8/7))), # 224 -> 256
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize(*mean_std)
                ])
    def __call__(self, x):
        return self.transform(x)