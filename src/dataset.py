import cv2
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import detectron2.data.transforms as T

import cv2
from read_video import faster_read_frame_at_index


def get_cropped_img(image, bbox, is_mask=False):
    crop_margin = 0.1

    size_x = image.shape[1]
    size_y = image.shape[0]

    x0, y0, x1, y1 = bbox

    dx = x1 - x0
    dy = y1 - y0

    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1

    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y

    if is_mask:
        crop = image[int(y0):int(y1), int(x0):int(x1)]
    else:
        crop = image[int(y0):int(y1), int(x0):int(x1), :]

    return crop

def inst_process(img, bbox, aspect_group, scale='S'):
    size_template = {'S': [(80, 320), (106, 320), (134, 320), (160, 320), (170, 298), (186, 280), (214, 266), (234, 234), 
                           (266, 214), (280, 186), (298, 170), (320, 160)],
                     'M': [(120, 480), (160, 480), (200, 480), (240, 480), (256, 448), (280, 420), (320, 400), (352, 352), 
                           (400, 320), (420, 280), (448, 256), (480, 240)],
                     'C': [(234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234),
                           (234, 234), (234, 234), (234, 234), (234, 234)]
                    }
    sizes = size_template[scale]

    inst = get_cropped_img(img, bbox, is_mask=False)
    inst = cv2.resize(inst, sizes[aspect_group])
    inst = cv2.cvtColor(inst, cv2.COLOR_BGR2GRAY)
    inst = cv2.cvtColor(inst, cv2.COLOR_GRAY2RGB)
    inst = standardize(transforms.functional.to_tensor(inst))
    return inst


class TaobaoImgBBoxInferSet(Dataset):
    def __init__(self, cfg, json_list):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self.json_list = json_list

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        dataset_dict = copy.deepcopy(self.json_list[idx])

        original_image = cv2.imread(dataset_dict["file_name"])
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]

        image = self.transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        dataset_dict['image'] = image

        return dataset_dict


standardize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class TaobaoMetricInferSet(Dataset):
    # (width, height)
    size_template = {'S': [(80, 320), (106, 320), (134, 320), (160, 320), (170, 298), (186, 280), (214, 266), (234, 234), 
                           (266, 214), (280, 186), (298, 170), (320, 160)],
                     'M': [(120, 480), (160, 480), (200, 480), (240, 480), (256, 448), (280, 420), (320, 400), (352, 352), 
                           (400, 320), (420, 280), (448, 256), (480, 240)],
                     'C': [(234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234), (234, 234),
                           (234, 234), (234, 234), (234, 234), (234, 234)]
                    }

    def __init__(self, json_list, scale='S'):
        self.json_list = json_list
        self.scale = scale
        self.sizes = self.size_template[scale]

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        dataset_dict = self.json_list[idx] # dict of single instance of an image/video

        if dataset_dict['type'] == 'image':
            image = cv2.imread(dataset_dict["file_name"])
        elif dataset_dict['type'] == 'video':
            image = faster_read_frame_at_index(dataset_dict["file_name"], dataset_dict["frame"])


        image = get_cropped_img(image, dataset_dict['bbox'], is_mask=False)
        image = cv2.resize(image, self.sizes[dataset_dict['aspect_group']])

        # if random.randint(0, 1) == 0:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = standardize(transforms.functional.to_tensor(image))
        return image