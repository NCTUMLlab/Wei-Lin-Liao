import  os
import  numpy  as np
import  torch
import  torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from image_helper import *
from parse_xml_annotations import *
from features import *
from reinforcement import *
from metrics import *
import logging
import time
import os
from Agent import PG
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def draw_bouding_box_1(annotation, img):
    new_img = Image.fromarray(img)
    draw = ImageDraw.Draw(new_img)
    length = len(annotation)
    annotation = np.array(annotation)
    for i in range(length):
        x_min = int(annotation[i,1])
        x_max = int(annotation[i,2])
        y_min = int(annotation[i,3])
        y_max = int(annotation[i,4])
        draw.line(((x_min, y_min), (x_max, y_min)), fill="red", width=3)
        draw.line(((x_min, y_min), (x_min, y_max)), fill="red", width=3)
        draw.line(((x_max, y_min), (x_max, y_max)), fill="red", width=3)
        draw.line(((x_min, y_max), (x_max, y_max)), fill="red", width=3)
    plt.figure()
    plt.imshow(new_img)

def get_annotation(offset, size_mask):
    annotation = np.zeros(5)
    annotation[3] = offset[0]
    annotation[4] = offset[0] + size_mask[0]
    annotation[1] = offset[1]
    annotation[2] = offset[1] + size_mask[1] 
    return annotation

print("load images")
path_voc = "../datas/VOCdevkit/VOC2007"
image_names = np.array(load_images_names_in_data_set('aeroplane_val', path_voc))
labels = load_images_labels_in_data_set('aeroplane_val', path_voc)
image_names_aero = []
for i in range(len(image_names)):
    if labels[i] == '1':
        image_names_aero.append(image_names[i])
image_names = image_names_aero
images = get_all_images(image_names, path_voc)
print("aeroplane_val image:%d" % len(image_names))

model_vgg = getVGG_16bn("../models")
model_vgg = model_vgg.cuda()

agent = PG(0.0002,0.90)
agent.load_model("./model_final/pg_VIME_agent_2")
exp_results_filename = "PG_VIME_2"

class_object = 1
steps = 5
res = []
res_step = []
res_annotations = []
for i in range(len(image_names)):
    image_name = image_names[i]
    image = images[i]
    
    # get use for iou calculation
    gt_annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
    original_shape = (image.shape[0], image.shape[1])
    classes_gt_objects = get_ids_objects_from_annotation(gt_annotation)
    gt_masks = generate_bounding_box_from_annotation(gt_annotation, image.shape)
    
    # the initial part
    region_image = image
    size_mask = image.shape
    region_mask = np.ones((image.shape[0], image.shape[1]))
    offset = (0, 0)
    history_vector = torch.zeros((4,6))
    state = get_state(region_image, history_vector, model_vgg)
    done = False
    
    # save the bounding box maked by agent
    annotations = []
    annotation = get_annotation(offset, size_mask)
    annotations.append(annotation)
    
    for step in range(steps):

            # Select action
            action = agent.select_action(Variable(state))+ 1
            policy_prob = agent.get_policy_prob(Variable(state))

            # Perform the action and observe new state
            if action == 6 or step == steps-1:
                next_state = None
                done = True
            else:
                offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,
                                                                   region_image, size_mask, action)
               
                #print(offset, size_mask, region_image.shape, region_mask.shape)
                annotation = get_annotation(offset, size_mask)
                annotations.append(annotation)
                history_vector = update_history_vector(history_vector, action)
                next_state = get_state(region_image, history_vector, model_vgg)

            # Move to the next state
            state = next_state
            if done:
                res_step.append(step)
                res_annotations.append((gt_annotation, annotations, image))
                break

                
    iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, class_object)

    if iou > 0.5:
        pos = 1
    else:
        pos = 0

    res.append((policy_prob[0,5], pos))


for i in range(0, 10):
    gt_annotation, annotation, image = res_annotations[i]
    draw_bouding_box_1(annotation, image)
    plt.savefig("./experiment_results/VIME_2/"+"Figure_"+str(i+1)+".png")

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
y_test = [x[1] for x in res]
y_score = [x[0] for x in res]
y_test = y_test[::-1]
y_score = y_score[::-1]
average_precision = average_precision_score(y_test, y_score)
precision, recall, _ = precision_recall_curve(y_test, y_score)

PR = [precision,recall]

#'''
with open("./experiment_data/"+exp_results_filename+"_PR_curve","wb") as fp:
        pickle.dump(PR,fp)
        print(precision)
        print(recall)
#'''

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1])
plt.xlim([0.0, 1])
plt.title('2-class Precision-Recall curve')
plt.savefig("./experiment_results/VIME_2/2-class Precision-Recall curve.png")
#plt.show()

res_step = np.array(res_step) + 1
#'''
with open("./experiment_data/"+exp_results_filename+"_num_region_hist","wb") as fp:
        pickle.dump(res_step,fp)
        print(res_step)
#'''
plt.hist(res_step)
plt.title('Number of reions analyze per object')
plt.xlabel('Number of regions')
plt.savefig("./experiment_results/VIME_2/Number of reions analyze per object.png")
#plt.show()
