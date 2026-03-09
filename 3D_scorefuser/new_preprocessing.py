import glob
import numpy as np
from skimage.transform import resize
import os
import argparse

def resize_img(img, img_size):
    img = resize(img, (img_size, img_size, img_size), mode='constant', cval=-1)
    img = (img * 2) - 1  # Normalize pixel values from 0-1 to -1 to 1
    return img

def main(input_data_dir, output_data_dir, img_size):
    SUFFIX = '.npy'

    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    img_list = list(glob.glob(input_data_dir + "*" + SUFFIX))
    for imgpath in img_list:
        imgname = imgpath.split('/')[-1]
        print('Processing:', imgname)
        if os.path.exists(output_data_dir + imgname):
            continue
        try:
            img = np.load(imgpath)
        except Exception as e:
            print(e)
            print("Image loading error:", imgname)
            continue
        try:
            img = resize_img(img, img_size)
        except Exception as e:
            print(e)
            print("Image resize error:", imgname)
            continue
        np.save(output_data_dir + imgname, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image resizing script')
    parser.add_argument('--input_data_dir', type=str, required=True,
                        help='Path to the input data directory')
    parser.add_argument('--output_data_dir', type=str, required=True,
                        help='Path to the output data directory')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Target image size (default: 256)')
    args = parser.parse_args()

    main(args.input_data_dir, args.output_data_dir, args.img_size)