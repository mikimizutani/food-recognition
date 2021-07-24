import torch
import os
from PIL import Image
import torch.utils.data
from pycocotools.coco import COCO
from torch import nn


class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(sorted(self.coco.getCatIds()))
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            if(xmin == xmax) or (ymin == ymax):
                continue
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        labels = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
            labels.append(coco_annotation[i]['category_id'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = [self.json_category_id_to_contiguous_id[c] for c in labels]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {"boxes": boxes, "labels": labels, "image_id": img_id, "area": areas, "iscrowd": iscrowd}

        if self.transforms is not None:
            img = self.transforms(img)
        print(my_annotation)
        return img, my_annotation

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        annotations = list()

        for b in batch:
            images.append(b[0])
            annotations.append(b[1])

        images = torch.stack(images, dim=0)

        return images, annotations # tensor (N, 3, 300, 300), 3 lists of N tensors each


ALPHA = 0.5
BETA = 0.5
GAMMA = 2


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky