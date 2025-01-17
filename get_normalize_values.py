import imutils.paths
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    IMAGES_PATH = r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset_pascal_voc\Evry_dataset_2024_pascal_voc\VOCdevkit\VOC\JPEGImages'

    d = {'r': {}, 'g': {}, 'b': {}}

    for index_image, image_path in tqdm(enumerate(list(imutils.paths.list_images(IMAGES_PATH)))):
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)



        for index, c in enumerate(d.keys()):
            c_channel = img_array[:, :, index].reshape(-1)
            c_std = np.std(c_channel)
            c_mean = np.mean(c_channel)
            c_max = np.max(c_channel)


            dict_image = {
                'std': c_std,
                'mean': c_mean,
                'max': c_max
            }

            if index_image == 0:
                d[c] = dict_image

            else:
                for k, v in dict_image.items():
                    d[c][k] += dict_image[k]

    max_pixel_value = 0
    for c in d.keys():
        if int(d[c]['max']) > max_pixel_value:
            max_pixel_value = int(d[c]['max'])

    list_rgb_mean = []
    list_rgb_std = []

    for c in d.keys():
        for values in ['mean', 'std']:
            d[c][values] /= len(list(imutils.paths.list_images(IMAGES_PATH)))
            d[c][values] /= max_pixel_value

            if values == 'mean':
                list_rgb_mean.append(d[c][values])

            if values == 'std':
                list_rgb_std.append(d[c][values])

    print(f'Max pixel value {max_pixel_value}')
    print(f'RGB mean {list_rgb_mean}')
    print(f'RGB std {list_rgb_std}')


