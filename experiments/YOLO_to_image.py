from torchvision import transforms
from model import YOLOv1
from PIL import Image
import argparse
import time
import os
import cv2
import torch

import matplotlib.pyplot as plt


# All BDD100K (dataset) classes and the corresponding class colors for drawing 
# the bounding boxes 
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                 "truck", "train", "other person", "bus", "car", "rider", 
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),
                  (128,255,0),(0,255,128),(255,0,127),(0,255,0),(0,0,255),
                  (127,0,255),(0,128,255),(128,128,128)]

# Argparse to apply YOLO algorithm to an image file from the console
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to the modell weights")
ap.add_argument("-t", "--threshold", default=0.5, 
                help="threshold for the confidence score of the bouding box prediction")
ap.add_argument("-ss", "--split_size", default=14, 
                help="split size of the grid which is applied to the image")
ap.add_argument("-nb", "--num_boxes", default=2, 
                help="number of bounding boxes which are being predicted")
ap.add_argument("-nc", "--num_classes", default=13, 
                help="number of classes which are being predicted")
ap.add_argument("-i1", "--input1", required=True, help="path to your input image")
ap.add_argument("-i2", "--input2", required=True, help="path to your input image 2")
ap.add_argument("-o1", "--output1", required=True, help="path to your output image")
ap.add_argument("-o2", "--output2", required=True, help="path to your output image 2")
args = ap.parse_args()





def get_iou(bb1, bb2):

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou






def detect():
    print("")
    print("##### YOLO OBJECT DETECTION FOR IMAGES #####")
    print("")   
    print("Loading the model")
    print("...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    print("Available devices: ", torch.cuda.device_count())
    torch.cuda.empty_cache()
    device = torch.device('cpu')
    model = YOLOv1(int(args.split_size), int(args.num_boxes), int(args.num_classes)).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Amount of YOLO parameters: " + str(num_param))
    print("...")
    print("Loading model weights")
    print("...")
    weights = torch.load(args.weights)
    model.load_state_dict(weights["state_dict"])
    model.eval()
    
    # Transform is applied to the input image
    # It resizes the image and converts it into a tensor
    transform = transforms.Compose([
        transforms.Resize((448,448), Image.NEAREST),
        transforms.ToTensor(),
        ])  
    
    print("Loading input image file")
    print("...")
    img1 = cv2.imread(args.input1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(args.input2, cv2.IMREAD_UNCHANGED)

    img2_copy = cv2.imread(args.input2, cv2.IMREAD_UNCHANGED)



    img1_width = img1.shape[1]
    img1_height = img1.shape[0]

    img2_width = img1.shape[1]
    img2_height = img1.shape[0]
    
    # Used to scale the bounding box predictions to the original input image
    # (448 is the dimension of the input image for the model)
    ratio_x1 = img1_width/448
    ratio_y1 = img1_height/448

    ratio_x2 = img2_width / 448
    ratio_y2 = img2_height / 448

    PIL_img1 = Image.fromarray(img1)
    PIL_img2 = Image.fromarray(img2)
    img1_tensor = transform(PIL_img1).unsqueeze(0).to(device)
    img2_tensor = transform(PIL_img2).unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time.time()
        output1 = model(img1_tensor) # Makes a prediction on the input image
        output2 = model(img2_tensor) # Makes a prediction on the input image
        curr_fps = int(1.0 / (time.time() - start_time)) # Prediction FPS
        print("FPS for YOLO prediction: " + str(curr_fps))
        print("")
        
    # Extracts the class index with the highest confidence scores
    corr_class1 = torch.argmax(output1[0,:,:,10:23], dim=2)
    corr_class2 = torch.argmax(output2[0,:,:,10:23], dim=2)

    bb1s = []
    bb2s = []

    # img1 bb
    for cell_h in range(output1.shape[1]):
        for cell_w in range(output1.shape[2]):
            # Determines the best bounding box prediction 
            best_box = 0
            max_conf = 0
            for box in range(int(args.num_boxes)):
                if output1[0, cell_h, cell_w, box*5] > max_conf:
                    best_box = box
                    max_conf = output1[0, cell_h, cell_w, box*5]
                
            # Checks if the confidence score is above the specified threshold
            if output1[0, cell_h, cell_w, best_box*5] >= float(args.threshold):
                # Extracts the box confidence score, the box coordinates and class
                confidence_score = output1[0, cell_h, cell_w, best_box*5]
                center_box = output1[0, cell_h, cell_w, best_box*5+1:best_box*5+5]
                best_class = corr_class1[cell_h, cell_w]
                    
                # Transforms the box coordinates into pixel coordinates
                centre_x = center_box[0]*32 + 32*cell_w
                centre_y = center_box[1]*32 + 32*cell_h
                width = center_box[2] * 448
                height = center_box[3] * 448
                    
                # Calculates the corner values of the bounding box
                x1 = int((centre_x - width/2) * ratio_x1)
                y1 = int((centre_y - height/2) * ratio_y1)
                x2 = int((centre_x + width/2) * ratio_x1)
                y2 = int((centre_y + height/2) * ratio_y1)

                bb1 = {}

                bb1['x1'] = x1
                bb1['x2'] = x2
                bb1['y1'] = y1
                bb1['y2'] = y2

                bb1s.append(bb1)

                # Draws the bounding box with the corresponding class color
                # around the object
                cv2.rectangle(img1, (x1,y1), (x2,y2), category_color[best_class], 1)
                cv2.rectangle(img2_copy, (x1,y1), (x2,y2), category_color[best_class], 1)
                # Generates the background for the text painted in the corresponding
                # class color and the text with the class label including the 
                # confidence score
                labelsize = cv2.getTextSize(category_list[best_class], 
                                            cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.rectangle(img1, (x1, y1-20), (x1+labelsize[0][0]+45,y1),
                              category_color[best_class], -1)
                cv2.putText(img1, category_list[best_class] + " " +
                            str(round(confidence_score.item(), 2)), (x1,y1-5), 
                            cv2.FONT_HERSHEY_DUPLEX , 0.5, (0,0,0), 1, cv2.LINE_AA)
                # Generates a small window in the top left corner which 
                # displays the current FPS for the prediction
                cv2.putText(img1, str(curr_fps) + "FPS", (25, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                cv2.rectangle(img2_copy, (x1, y1-20), (x1+labelsize[0][0]+45,y1),
                              category_color[best_class], -1)
                cv2.putText(img2_copy, category_list[best_class] + " " +
                            str(round(confidence_score.item(), 2)), (x1,y1-5),
                            cv2.FONT_HERSHEY_DUPLEX , 0.5, (0,0,0), 1, cv2.LINE_AA)
                # Generates a small window in the top left corner which
                # displays the current FPS for the prediction
                cv2.putText(img2_copy, str(curr_fps) + "FPS", (25, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


        # img2 bb
    for cell_h in range(output2.shape[1]):
        for cell_w in range(output2.shape[2]):
            # Determines the best bounding box prediction
            best_box = 0
            max_conf = 0
            for box in range(int(args.num_boxes)):
                if output2[0, cell_h, cell_w, box * 5] > max_conf:
                    best_box = box
                    max_conf = output2[0, cell_h, cell_w, box * 5]

            # Checks if the confidence score is above the specified threshold
            if output2[0, cell_h, cell_w, best_box * 5] >= float(args.threshold):
                # Extracts the box confidence score, the box coordinates and class
                confidence_score = output2[0, cell_h, cell_w, best_box * 5]
                center_box = output2[0, cell_h, cell_w, best_box * 5 + 1:best_box * 5 + 5]
                best_class = corr_class2[cell_h, cell_w]

                # Transforms the box coordinates into pixel coordinates
                centre_x = center_box[0] * 32 + 32 * cell_w
                centre_y = center_box[1] * 32 + 32 * cell_h
                width = center_box[2] * 448
                height = center_box[3] * 448

                # Calculates the corner values of the bounding box
                x1 = int((centre_x - width / 2) * ratio_x2)
                y1 = int((centre_y - height / 2) * ratio_y2)
                x2 = int((centre_x + width / 2) * ratio_x2)
                y2 = int((centre_y + height / 2) * ratio_y2)

                bb2 = {}

                bb2['x1'] = x1
                bb2['x2'] = x2
                bb2['y1'] = y1
                bb2['y2'] = y2

                bb2s.append(bb2)

                # Draws the bounding box with the corresponding class color
                # around the object
                cv2.rectangle(img2, (x1, y1), (x2, y2), category_color[best_class], 1)
                # Generates the background for the text painted in the corresponding
                # class color and the text with the class label including the
                # confidence score
                labelsize = cv2.getTextSize(category_list[best_class],
                                            cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.rectangle(img2, (x1, y1 - 20), (x1 + labelsize[0][0] + 45, y1),
                              category_color[best_class], -1)
                cv2.putText(img2, category_list[best_class] + " " +
                            str(round(confidence_score.item(), 2)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # Generates a small window in the top left corner which
                # displays the current FPS for the prediction
                cv2.putText(img2, str(curr_fps) + "FPS", (25, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    min_bb = bb1s if len(bb1s) < len(bb2s) else bb2s
    max_bb = bb2s if len(bb1s) < len(bb2s) else bb1s

    iou_total = 0
    for i in range(len(min_bb)):
        max_iou = 0
        for j in range(len(max_bb)):
            iou = get_iou(min_bb[i], max_bb[j])
            if iou > max_iou:
                max_iou = iou
        print(max_iou)
        iou_total += max_iou

    print("total: ", iou_total)
    print("iou: ", iou_total / len(max_bb))

    iou = iou_total / len(max_bb)

    print("#######  IOU: ", iou, " #######")

    f, ax = plt.subplots(3)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[2].imshow(img2_copy)
    plt.show()

    cv2.imwrite(args.output1, img1) # Stores the image with the predictions in a new file
    cv2.imwrite(args.output2, img2) # Stores the image with the predictions in a new file

    return img1, img2, img2_copy, iou

if __name__ == '__main__':
    detect()