import imgaug.augmenters as iaa
import os
import argparse
import cv2
import secrets

parser = argparse.ArgumentParser(description='Image Data Augmentation using imgaug')

def arg_directory(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'`{path}` is not valid')

parser.add_argument('--path',
                    type = arg_directory,
                    help = 'Directory to Dataset',
                    default = None
)

parser.add_argument('--output',
                    type = arg_directory,
                    help = 'Directory to save Outputs',
                    default = None
)

args = parser.parse_args()

seq = iaa.Sequential([
    iaa.SomeOf((0, 7), 
    [iaa.Grayscale(alpha = (0.5, 1.0)),
    iaa.Sometimes(0.5, iaa.Add(value=(-40, 40), per_channel=False)),
    iaa.Sometimes(0.5, iaa.Multiply(mul = (0.8, 1.2))),
    iaa.Sometimes(0.3, iaa.GammaContrast(gamma=(0.3, 2.0))),
    # iaa.Fliplr(p = 0.5),
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma = (0.0, 1.0))),
    iaa.Sometimes(0.3, iaa.MotionBlur(k = 3, random_state=True)),
    iaa.Sometimes(1.0, iaa.CropAndPad(percent=(-0.1, 0.1), keep_size=True)),
    iaa.Sometimes(0.5, iaa.Affine(scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)}, rotate=(-30, 30), order=[0, 1]))
    ])
])

for im_file in os.listdir(args.path):
    image = cv2.imread(os.path.join(args.path, im_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(100):
        image_aug = seq(image = image)
        filename = str(secrets.token_hex(6)) + '.jpg'
        cv2.imwrite(os.path.join(args.output, filename), image_aug)