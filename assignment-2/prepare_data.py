from torchvision import datasets, models, transforms
import os

def fetch_data(data_dir, splits):
    ''' Load data '''
    ''' Pytorch needs images to have minibatches 3 x H x W with (H,W) >= (224,224).
        So images are converted to [0,1] range and normalized with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ]),
    }

    dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in splits}
    return dataset
