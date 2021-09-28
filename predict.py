import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os, random
import torch
import torchvision
from train import get_model_instance_segmentation, get_cat_dict, get_binary_mask, nms
from classes import FoodDataset


if __name__ == '__main__':
    #Provide folder that contains images
    test_path = './data/test/images'
    test_json = './data/test/annotations.json'
    validation_data = FoodDataset(root=test_path, annotation=test_json, transforms=None)
    path = os.path.join(test_path, random.choice(os.listdir(test_path)))
    #Or pass a specific path
    #path = "./data/test/images/025585.jpg"

    model_path = './food_net.pth'
    model = get_model_instance_segmentation()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    img = Image.open(path)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((512,512)),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img_tr = list([transform(img)])
    output = model(img_tr)
    pred_labels = output[0].get('labels').tolist()
    pred_scores = output[0].get('scores').tolist()
    pred_masks = output[0].get('masks').tolist()
    pred_bbox = output[0].get('boxes').tolist()
    class_dict = get_cat_dict(test_json, validation_data)
    class_labels = [class_dict[x] for x in pred_labels]
    if len(pred_labels) > 1:
        keep = nms(pred_bbox, pred_scores)
        pred_labels = [pred_labels[i] for i in keep]
        pred_bbox = [pred_bbox[i] for i in keep]
        pred_masks = [pred_masks[i] for i in keep]
        pred_scores = [pred_scores[i] for i in keep]
    fig, ax = plt.subplots()
    ax.imshow(img.resize((512,512)))
    currentAxis = plt.gca()
    for i in range(0,len(pred_scores)):
        currentAxis.add_patch(Rectangle((pred_bbox[i][0], pred_bbox[i][1]), pred_bbox[i][2]-pred_bbox[i][0],pred_bbox[i][3]-pred_bbox[i][1], alpha=1, fill=None))
        currentAxis.annotate(class_labels[i] + " " + str(round(pred_scores[i],2)), (pred_bbox[i][0], pred_bbox[i][1]))
        ax.imshow(get_binary_mask(pred_masks[i][0]), alpha=0.2, interpolation='none')

    plt.show()
