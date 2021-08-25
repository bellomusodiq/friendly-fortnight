import torchvision.transforms as transforms
import numpy as np


transformers = {
    'center_crop': transforms.CenterCrop((310)),
    'random_crop': transforms.RandomCrop((310)),
    'color_jitter': transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.2),
    'gray_scale': transforms.Grayscale(),
    'affine': transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.8, 1.2), shear=10),
    'horizontal_flip': transforms.RandomHorizontalFlip(p=1),
    'vertical_flip': transforms.RandomVerticalFlip(p=1),
    'perspective': transforms.RandomPerspective(p=1),
    'rotation': transforms.RandomPerspective(p=1),
    'gaussian_blur': transforms.GaussianBlur(51),
    'invert': transforms.RandomInvert(p=1),
    'sharpness': lambda x: transforms.functional.adjust_sharpness(x, np.random.uniform(0,2)),
    'random_erasing': transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomErasing(p=1, scale=(0.1, 0.1), 
                                ratio=(0.2, 0.2), value=0, inplace=False),
                            transforms.ToPILImage()
                        ]),
}


def transform_image(image, transforms):
    transformed_image = []
    for transform in transforms:
        if transformers.get(transform):
            img = transformers[transform](image)
            transformed_image.append({'transform': transform, 'image': img})
    return transformed_image
