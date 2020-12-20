# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:31:19 2020

@author: JAY KISHAN OMAR
"""

from scipy.spatial import distance as dist
import cv2
import numpy as np
import imutils


Min_confi = 0.3
NMS_thres = 0.3

Use_GPU = True
Min_distance = 50

net = cv2.dnn.readNet("yolov3_my.weights", "yolov3_my.cfg")
classes = []
if Use_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layers = net.getLayerNames()
output_layers = [layers[i[0]-1] for i in net.getUnconnectedOutLayers()]
print(layers)
print(output_layers)
print(net.getUnconnectedOutLayers())
print(layers[199], layers[226], layers[253])


def detecting_people(frame, net, output_layers, personIndex = 0):
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    boxes = []
    confidences = []
    centroids = []
    class_ids = []
    results = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5 and class_id == personIndex:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centroids.append((center_x, center_y))
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, Min_confi, NMS_thres)
    if len(indexes)>0:
        for i in indexes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = (confidences[i], (x,y,x+w,y+h), centroids[i])
            results.append(res)
            
    return results

filepath = r"pedestrians.mp4"
def get_file(user_wish):
    if (user_wish):
        return filepath
    else:
        return 0
    
    
cam = cv2.VideoCapture(get_file(False))
writer = None
count = 0

while True:
    (grabbed, frame) = cam.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width = 700)
    results = detecting_people(frame, net, output_layers, personIndex = 0)
    
    violate = set()
    if len(results)>=2:
        
        centroids = np.array([res[2] for res in results])
        D = dist.cdist(centroids, centroids, metric = "euclidean")
        for i in range(0,D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i, j]< Min_distance:
                    violate.add(i)
                    violate.add(j)
                    
                
                    
    for (i, (r, box, centroid)) in enumerate(results):
        (upper_x,upper_y,lower_x,lower_y) = box
        (cx, cy) = centroid
        color = (0,255,0)
        
        if i in violate:
            color = (0,0,255)
        cv2.rectangle(frame, (upper_x, upper_y), (lower_x,lower_y), color, 2)
        cv2.circle(frame, (cx,cy), 10, color, 1)
        
    text = "Social Distancing Violation: {}".format(len(violate))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, frame.shape[0]-25), font, 0.85, (0,0,255), 3)
    text2 = "Frames = {}".format(count)
    cv2.putText(frame, text2, (10, frame.shape[0]-60), font, 0.5, (0,255,0), 2)
    
    cv2.imshow("video capturing", frame)
    count = count+1
        
    keys = cv2.waitKey(20)
    if keys == ord('q'):
        cam.release()
        cv2.destroyAllWindows()    
            
        
