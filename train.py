import sys
import torch
import statistics
from pycocotools.coco import COCO
from classes import FoodDataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
from sklearn.metrics import jaccard_score


def to_tensor():
    tensors = [torchvision.transforms.Resize((512,512)),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return torchvision.transforms.Compose(tensors)


#From https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def get_model_instance_segmentation():
    num_classes = 274
        # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def train(model, training_dataloader, device, optimizer):
    model.train()
    for imgs, annotations in training_dataloader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in list(loss_dict.values()))
        optimizer.zero_grad()
        losses.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
    return model, losses


def nms(bbox, scores): #Non maxima suppression of predictions
    length = len(bbox)
    keep = set()
    discard = set()
    for i in range(length):
        if i in discard:
            continue
        if scores[i] < 0.5:
            discard.add(i)
            continue
        for j in range(length):
            if (i == j) or (j in discard):
                continue
            iou = bbox_iou(bbox[i], bbox[j])
            if iou >= 0.5:
                keep.add(min(i,j))
                discard.add(max(i,j))
        if i not in discard:
            keep.add(i)
    return keep


def analyze(pred_labels, gt_labels, pred_bbox, gt_bbox, pred_masks, gt_masks, bbox_stats, mask_stats):
    for i in range(0, len(pred_labels)):
        if pred_labels[i] in gt_labels:
            for j in range(0, len(gt_labels)):
                if pred_labels[i] == gt_labels[j]:
                    bbox_stats[pred_labels[i]] = analyze_bbox(pred_bbox[i], gt_bbox[j], bbox_stats.get(pred_labels[i]))
                    mask_stats[pred_labels[i]] = analyze_mask(pred_masks[i][0], gt_masks[j], mask_stats.get(pred_labels[i]))
                    continue
        else:
            bbox_stats[pred_labels[i]]['FP'] += 1
            mask_stats[pred_labels[i]]['FP'] += 1
    for label in gt_labels:
        if not label in pred_labels:
            bbox_stats[label]['FN'] += 1
            mask_stats[label]['FN'] += 1
    return bbox_stats, mask_stats

#Adapted from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bbox_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    predArea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    gtArea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return interArea / float(predArea + gtArea - interArea)


def analyze_bbox(pred, gt, stats):
    iou = bbox_iou(pred, gt)
    if iou >= 0.5:
        stats['TP'] += 1
    else:
        stats['FP'] += 1
    return stats


def get_binary_mask(mask):
    for i in range(0, len(mask)):
       mask[i] = [round(m) for m in mask[i]]
    return mask


def analyze_mask(pred, gt, stats):
    pred = get_binary_mask(pred)
    iou = jaccard_score(pred, gt, average='micro', zero_division=0.0)
    if iou >= 0.5:
        stats['TP'] += 1
    else:
        stats['FP'] += 1
    return stats


def print_stats(stats, cat_dictionary, type):
    precision = {}
    recall = {}
    iou = {}
    for classname, metrics in stats.items():
        classname = cat_dictionary[classname]
        if metrics['TP'] == 0:
            precision[classname] = 0
            recall[classname] = 0
            iou[classname] = 0
        else:
            precision[classname] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
            recall[classname] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
            iou[classname] = metrics['TP'] / (metrics['TP'] + metrics['FN'] + metrics['FP'])
    print(type + ' Precision: ' + str(precision))
    print(type + ' Recall: ' + str(recall))
    print(type + ' IOU: ' + str(iou))
    print(type + ' Mean IOU: ' + str(statistics.mean(iou.values())))


def test(model, validation_dataloader, device, cat_dictionary):
    model.eval()

    bbox_stats = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(1, 274)}
    mask_stats = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(1, 274)}

    with torch.no_grad():
        for imgs, annotations in validation_dataloader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            output = model(imgs)

            for i in range(0, len(output)):
                pred_labels = output[i].get('labels').tolist()
                gt_labels = annotations[i].get('labels').tolist()
                pred_masks = output[i].get('masks').tolist()
                gt_masks = annotations[i].get('masks').tolist()
                pred_bbox = output[i].get('boxes').tolist()
                gt_bbox = annotations[i].get('boxes').tolist()
                scores = output[i].get('scores').tolist()

                if len(pred_labels) > 1:
                    keep = nms(pred_bbox, scores)
                    pred_labels = [pred_labels[i] for i in keep]
                    pred_bbox = [pred_bbox[i] for i in keep]
                    pred_masks = [pred_masks[i] for i in keep]
                bbox_stats, mask_stats = analyze(pred_labels, gt_labels, pred_bbox, gt_bbox, pred_masks, gt_masks, bbox_stats, mask_stats)
    print_stats(bbox_stats, cat_dictionary, 'Bounding Box')
    print_stats(mask_stats, cat_dictionary, 'Mask')


def get_cat_dict(json, dataset):
    coco = COCO(json)
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    cat_ids = [cat['id'] for cat in cats]
    cat_ids = [dataset.json_category_id_to_continuous_id[cat] for cat in cat_ids]
    cat_dictionary = dict(zip(cat_ids, cat_names))
    return cat_dictionary


if __name__ == '__main__':
    training_path = './data/train/images'
    training_json = './data/train/annotations.json'
    test_path = './data/test/images'
    test_json = './data/test/annotations.json'

    num_epochs = 50

    training_data = FoodDataset(root=training_path, annotation=training_json, transforms=to_tensor())
    validation_data = FoodDataset(root=test_path, annotation=test_json, transforms=to_tensor())
    training_dataloader = DataLoader(training_data, batch_size=10, shuffle=True, collate_fn=training_data.collate_fn)
    validation_dataloader = DataLoader(validation_data, batch_size=10, shuffle=True, collate_fn=validation_data.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0001)

    cat_dictionary = get_cat_dict(test_json, validation_data)

    PATH = './food_net.pth'
    for i in range(1, num_epochs+1):
       model, losses = train(model, training_dataloader, device, optimizer)
       print(f'Iteration: {i}/{num_epochs}, Loss: {losses}')
       test(model, validation_dataloader, device, cat_dictionary)
    torch.save(model.state_dict(), PATH)
    print('Training finished')
