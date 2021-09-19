# # -*- coding: utf-8 -*-
# """
# Created on Wed Aug 11 19:42:42 2021

# @author: chakr
# """

# import imutils
# from imutils.object_detection import non_max_suppression
# import cv2
# import argparse
# import numpy as np
# import time

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, help="image")
# ap.add_argument("-east", "--east", type=str, default="frozen_east_text_detection.pb", help="EAST path")
# ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="roi prob") #0.5
# ap.add_argument("-w", "--width", type=int, default=320,	help="w.32") #32
# ap.add_argument("-e", "--height", type=int, default=320,help="h.32") #32
# args = vars(ap.parse_args())

# image=cv2.imread(args["image"])
# orig=image.copy()
# (H, W) = image.shape[:2]

# (newW, newH) = (args["width"], args["height"])
# rW = W / float(newW)
# rH = H / float(newH)


# image = cv2.resize(image, (newW, newH))
# (H, W) = image.shape[:2]
# layerNames = [	"feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
# print("[INFO] loading EAST text detector...")
# net = cv2.dnn.readNet(args["east"])
# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
# start = time.time()
# net.setInput(blob)
# (scores, geometry) = net.forward(layerNames)
# end = time.time()
# print("[INFO] text detection took {:.6f} seconds".format(end - start))

# (numRows, numCols) = scores.shape[2:4]
# rects = []
# confidences = []

# for y in range(0, numRows):
#     scoresData = scores[0, 0, y]
#     xData0 = geometry[0, 0, y]
#     xData1 = geometry[0, 1, y]	
#     xData2 = geometry[0, 2, y]	
#     xData3 = geometry[0, 3, y]
#     anglesData = geometry[0, 4, y]
    
#     for x in range(0, numCols):
#         if (scoresData[x] < args["min_confidence"]):
#             continue
#         (offsetX, offsetY) = (x * 4.0, y * 4.0)
#         angle = anglesData[x]
#         cos = np.cos(angle)
#         sin = np.sin(angle)
#         h = xData0[x] + xData2[x]
#         w = xData1[x] + xData3[x]
#         endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#         endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#         startX = int(endX - w)
#         startY = int(endY - h)
#         rects.append((startX, startY, endX, endY))
#         confidences.append(scoresData[x])

# boxes = non_max_suppression(np.array(rects), probs=confidences)

# for (startX, startY, endX, endY) in boxes:
#     startX = int(startX * rW)
#     startY = int(startY * rH)
#     endX = int(endX * rW)
#     endY = int(endY * rH)
#     cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
# cv2.imshow("Text Detection", orig)
# cv2.waitKey(0)


import cv2 as cv
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

parser.add_argument('--model', default="frozen_east_text_detection.pb")

parser.add_argument('--width', type=int, default=320)

parser.add_argument('--height',type=int, default=320)

parser.add_argument('--thr',type=float, default=0.5)

parser.add_argument('--nms',type=float, default=0.4)
parser.add_argument('--device', default="cpu", help="Device to inference on")


args = parser.parse_args()


def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

           
            if(score < scoreThresh):
                continue
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    return [detections, confidences]

if __name__ == "__main__":
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model

    net = cv.dnn.readNet(model)
    if args.device == "cpu":
        net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")


    kWinName = "Text Detection"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    cap = cv.VideoCapture(args.input if args.input else 0)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        net.setInput(blob)
        output = net.forward(outputLayers)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        for i in indices:
 
            vertices = cv.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
      # Put efficiency information
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Display the frame
        cv.imshow(kWinName,frame)
        cv.imwrite("output.png",frame)