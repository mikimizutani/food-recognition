import os
import sys

import torch
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from classes import FoodDataset
from classes import NeuralNetwork
from classes import FocalTverskyLoss
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
from engine import train_one_epoch, evaluate
import utils


def to_tensor():
    tensors = [torchvision.transforms.Resize((400,400)), torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(tensors)


#Method to view image with the categories
def visualise_annotations(coco):
    # nms = set([cat['supercategory'] for cat in cats])
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    image = mpimg.imread(os.path.join('./dataset/train/images/', str(img['file_name'])))
    plt.axis('off')
    plt.imshow(image)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


def train(model, device, train_loader, optimizer):
    model.train()
    y_true = []
    y_pred = []
    for i in train_loader:

        # LOADING THE DATA IN A BATCH
        data, target = i

        # MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data, target = data.to(device), target.to(device)

        # FORWARD PASS
        output = model(data.float())
        loss = FocalTverskyLoss(output)
        #loss = criterion(output, target.unsqueeze(1))

        # BACKWARD AND OPTIMIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # PREDICTIONS
        pred = np.round(output.detach())
        target = np.round(target.detach())
        y_pred.extend(pred.tolist())
        y_true.extend(target.tolist())

        print("Accuracy on training set is",
              accuracy_score(y_true, y_pred))


def val(model, device, test_loader):
    # model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []

    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i in test_loader:
            # LOAD THE DATA IN A BATCH
            data, target = i

            # moving the tensors to the configured device
            data, target = data.to(device), target.to(device)

            # the model on the data
            output = model(data.float())

            # PREDICTIONS
            pred = np.round(output)
            target = target.float()
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())

    print("Accuracy on test set is", accuracy_score(y_true, y_pred))
    print("***********************************************************")


def accuracy_score(true, pred):
    #TODO
    return


def get_model_instance_segmentation(num_classes):
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


if __name__ == '__main__':
    training_path = './dataset/train/images'
    training_json = './dataset/train/annotations.json'
    test_path = './dataset/val/images'
    test_json = './dataset/val/annotations.json'

    # dataDir = './dataset'
    # dataType = 'train'
    # annFile = './dataset/train/annotations.json'.format(dataDir, dataType)
    # coco = COCO(annFile)
    #
    # cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))
    # print(len(nms))

    #visualise_annotations()
    num_epochs = 100

    training_data = FoodDataset(root=training_path, annotation=training_json, transforms=to_tensor())
    validation_data = FoodDataset(root=test_path, annotation=test_json, transforms=to_tensor())
    training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 273
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, training_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        val(model, device, validation_dataloader)

    print("That's it!")