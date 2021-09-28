import torch
import os
from PIL import Image
import torch.utils.data
from pycocotools.coco import COCO
import torch.nn.functional
import skimage.transform

#Adapted from https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
# and from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.json_category_id_to_continuous_id = {
            v: i+1 for i, v in enumerate(sorted(self.coco.getCatIds()))
        }
        self.continuous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_continuous_id.items()
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
        width, height = img.size
        width_scale = 512 / width
        height_scale = 512 / height
        for i in range(num_objs):
            xmin = (coco_annotation[i]['bbox'][1]) * width_scale
            ymin = (coco_annotation[i]['bbox'][0]) * height_scale
            xmax = xmin + (coco_annotation[i]['bbox'][3] * width_scale)
            ymax = ymin + (coco_annotation[i]['bbox'][2] * height_scale)
            if(xmin == xmax) or (ymin == ymax):
                continue
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img_id = torch.tensor([img_id])
        areas = []
        labels = []
        masks = []

        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
            cat_id = self.json_category_id_to_continuous_id[coco_annotation[i]['category_id']]
            labels.append(cat_id)
            mask = coco.annToMask(coco_annotation[i])
            mask = skimage.transform.resize(mask, (512,512), order=0, preserve_range=True, anti_aliasing=False)
            masks.append(mask)

        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        img = self.transforms(img)

        # Annotation is in dictionary format
        my_annotation = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": img_id, "area": areas, "iscrowd": iscrowd}
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