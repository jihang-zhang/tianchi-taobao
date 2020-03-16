import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
import torch.nn.functional as F

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from network import Net
from prepare_json import get_vid_dicts
from read_video import faster_read_frame_at_index
from dataset import inst_process

DEBUG = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_vid(cfg, metric_net, dataset, bbox_scale='S', frames=[40, 120, 200, 280, 360]):
    metric_net.to(DEVICE)
    metric_net.eval()

    aspect_template = np.array([0.25, 0.33333333, 0.41666667, 0.5, 0.57142857, 0.66666667, 0.8, 1., 1.25, 1.5, 1.75, 2.])

    result = []
    predictor = DefaultPredictor(cfg)

    for image_dict in tqdm(dataset):
        for frame in frames:
            img = faster_read_frame_at_index(image_dict['file_name'], frame)
            outputs = predictor(img)
            scores = outputs['instances'].get_fields()['scores']
            pred_boxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy().astype(int).tolist()

            with torch.no_grad():
                for bbox, score in zip(pred_boxes, scores):
                    inst_dict = copy.deepcopy(image_dict)
                    inst_dict['bbox'] = bbox
                    inst_dict['score'] = float(score)
                    inst_dict['frame'] = frame
                    inst_dict['bbox_aspect'] = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
                    inst_dict['aspect_group'] = int(np.abs(inst_dict['bbox_aspect'] - aspect_template).argmin())

                    inst = inst_process(img, bbox, inst_dict['aspect_group'], scale=bbox_scale).to(DEVICE)
                    feat = F.normalize(metric_net(inst.unsqueeze(0)))[0].detach().cpu().numpy().tolist()
                    inst_dict['feat'] = feat

                    result.append(inst_dict)

    return result

def get_pred_vid():
    cfg = get_cfg()
    cfg.merge_from_file('../output/one_cls_faster_rcnn_R_50_FPN_deconv/config.yaml')
    cfg.MODEL.WEIGHTS = '../output/one_cls_faster_rcnn_R_50_FPN_deconv/model_0044999.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    vid_set_part1 = get_vid_dicts('/tcdata/test_dataset_3w/test_dataset_part1/video')
    vid_set_part2 = get_vid_dicts('/tcdata/test_dataset_3w/test_dataset_part2/video')

    dataset = vid_set_part1 + vid_set_part2

    if DEBUG:
        dataset = dataset[:400]

    metric_net = Net(num_classes=29841, feat_dim=512, cos_layer=True, dropout=0., image_net='resnet50', pretrained=False)
    checkpoint = torch.load('../output/arcface_R_50/best_14.pt')
    metric_net.load_state_dict(checkpoint['model_state_dict'])

    inst_ds = infer_vid(cfg, metric_net, dataset, bbox_scale='S', frames=[40, 120, 200, 280, 360])

    return inst_ds

def main():
    cfg = get_cfg()
    cfg.merge_from_file('../output/one_cls_faster_rcnn_R_50_FPN_deconv/config.yaml')
    cfg.MODEL.WEIGHTS = '../output/one_cls_faster_rcnn_R_50_FPN_deconv/model_0044999.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    vid_set_part1 = get_vid_dicts('/tcdata/test_dataset_3w/test_dataset_part1/video')
    vid_set_part2 = get_vid_dicts('/tcdata/test_dataset_3w/test_dataset_part2/video')

    dataset = vid_set_part1 + vid_set_part2

    if DEBUG:
        dataset = dataset[:400]

    metric_net = Net(num_classes=29841, feat_dim=512, cos_layer=True, dropout=0., image_net='resnet50', pretrained=False)
    checkpoint = torch.load('../output/arcface_R_50/best_14.pt')
    metric_net.load_state_dict(checkpoint['model_state_dict'])

    inst_ds = infer_vid(cfg, metric_net, dataset, bbox_scale='S', frames=[40, 120, 200, 280, 360])
    with open('../output/inference/pred_vid.json', 'w') as f:
        json.dump(inst_ds, f)

if __name__ == "__main__":
    main()