# use following code in console to check the whole dataset of ground truths
# import fiftyone as fo
# import fiftyone.zoo as foz
# dataset = foz.load_zoo_dataset("open-images-v6", "validation")
# session = fo.launch_app(dataset, port=5151)
# ---------------------------------------------------------------------

import sys
#print 'number of arguments:', len(sys.argv), 'arguments.'
#print 'argument list:', str(sys.argv)
sys.path.append("C:/Users/ja/PycharmProjects/YOLOaddition/venv/Lib/site-packages/")

import cv2
import pandas as pd
import math
import random


def read3images(location, imgName, nr):
    image = cv2.imread(location + "/" + nr + "/" + imgName, 1)
    rImage = cv2.imread(location + "/" + nr + "/data/rotated.jpg", 1)
    original = pd.read_csv(location + "/" + nr + "/data/original.csv")
    mirror = pd.read_csv(location + "/" + nr + "/data/mirror.csv")
    rotated = pd.read_csv(location + "/" + nr + "/data/rotated.csv")
    return image, rImage, original, mirror, rotated


def read2imagesM(location, imgName, nr):
    image = cv2.imread(location + "/" + nr + "/" + imgName, 1)
    original = pd.read_csv(location + "/" + nr + "/data/original.csv")
    mirror = pd.read_csv(location + "/" + nr + "/data/mirror.csv")
    return image, original, mirror


def read2imagesR(location, imgName, nr):
    image = cv2.imread(location + "/" + nr + "/" + imgName, 1)
    rImage = cv2.imread(location + "/" + nr + "/data/rotated.jpg", 1)
    original = pd.read_csv(location + "/" + nr + "/data/original.csv")
    rotated = pd.read_csv(location + "/" + nr + "/data/rotated.csv")
    return image, rImage, original, rotated


# transforms bbox data from 0~1 range to pixels
def transformOrgCoordinates(original, width, height):
    for i in range(original.shape[0]):
        original.loc[i, "x"] = original.loc[i, "x"] * width
        original.loc[i, "y"] = original.loc[i, "y"] * height
        original.loc[i, "w"] = original.loc[i, "w"] * width
        original.loc[i, "h"] = original.loc[i, "h"] * height
        # print(original.loc[i, "class"], "\t",  original.loc[i, "x"], original.loc[i, "y"])


# transforms bbox data from 0~1 range to pixels and makes them corresponding to those from original image
def transformMirrCoordinates(mirror, width, height):
    for i in range(mirror.shape[0]):
        mirror.loc[i, "x"] = width - (mirror.loc[i, "x"] * width)
        mirror.loc[i, "y"] = mirror.loc[i, "y"] * height
        mirror.loc[i, "w"] = mirror.loc[i, "w"] * width
        mirror.loc[i, "h"] = mirror.loc[i, "h"] * height
        # print(mirror.loc[i, "class"], "\t",  mirror.loc[i, "x"], mirror.loc[i, "y"])


# transforms bbox data from 0~1 range to pixels and makes them corresponding to those from original image
def transformRotCoordinates(rotated, rWidth, rHeight, marginW, marginH):

    for i in range(rotated.shape[0]):
        x = rotated.loc[i, "x"] * rWidth
        y = rotated.loc[i, "y"] * rHeight
        y = rHeight - y
        # print(rotated.loc[i, "class"], "\t", x, y)
        x = x - (rWidth / 2)
        y = y - (rHeight / 2)
        tg = y / x
        fi = math.fabs(math.atan(tg))  # returns from -pi/2 to pi/2 rad
        if x < 0 and y > 0:
            fi = -fi + math.pi
        elif x < 0 and y < 0:
            fi = fi + math.pi
        elif x > 0 and y < 0:
            fi = -fi + (2 * math.pi)
        # print(math.degrees(fi))
        r = math.sqrt((x ** 2) + (y ** 2))
        x = r * math.cos(fi + math.radians(5))
        y = r * math.sin(fi + math.radians(5))
        x = x + (rWidth / 2)
        y = y + (rHeight / 2)
        y = rHeight - y
        x = x - (marginW / 2)
        y = y - (marginH / 2)
        rotated.loc[i, "x"] = x
        rotated.loc[i, "y"] = y
        rotated.loc[i, "w"] = rotated.loc[i, "w"] * rWidth
        rotated.loc[i, "h"] = rotated.loc[i, "h"] * rHeight
        # print("\t", x, y)


def removeUnclassified(bigTable):
    for i in range(bigTable.shape[0]):
        if bigTable.loc[i, "class"] == -1:
            bigTable.drop([i], inplace=True)
        else:
            break
    bigTable.reset_index(drop=True, inplace=True)


def removeLowProbabilities(bigTable, probThresh):
    for i in range(bigTable.shape[0]):
        if bigTable.loc[i, "probability"] < probThresh:
            bigTable.drop([i], inplace=True)
    bigTable.reset_index(drop=True, inplace=True)


# uses standard average
def calculateResults(bigTable, accuracy):
    results = pd.DataFrame(columns=["hits", "probability", "class", "x", "y", "w", "h"])
    x2 = 0
    y2 = 0
    w2 = 0
    h2 = 0
    prob2 = 0
    count = 1
    cl = -1
    reset = 0
    for i in range(bigTable.shape[0]):
        cl = bigTable.loc[i, "class"]
        tempX = bigTable.loc[i, "x"]
        tempY = bigTable.loc[i, "y"]
        tempW = bigTable.loc[i, "w"]
        tempH = bigTable.loc[i, "h"]
        tempProb = bigTable.loc[i, "probability"]
        if i < (bigTable.shape[0] - 1):
            nextCl = bigTable.loc[i + 1, "class"]
        if i == 0 or reset == 1:
            x2 = tempX
            y2 = tempY
            w2 = tempW
            h2 = tempH
            prob2 = tempProb
            count = 1
            reset = 0
        else:
            if cl == nextCl:
                if (tempX > ((x2 / count) - accuracy)) and (tempX < ((x2 / count) + accuracy)) and (
                        tempY > ((y2 / count) - accuracy)) and (tempY < ((y2 / count) + accuracy)):
                    x2 = x2 + tempX
                    y2 = y2 + tempY
                    w2 = w2 + tempW
                    h2 = h2 + tempH
                    prob2 = prob2 + tempProb
                    count = count + 1
                else:  # for next object of the same class
                    results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [count,
                                                                                                     (prob2 / count),
                                                                                                     cl, (x2 / count),
                                                                                                     (y2 / count),
                                                                                                     (w2 / count),
                                                                                                     (h2 / count)]
                    x2 = tempX
                    y2 = tempY
                    w2 = tempW
                    h2 = tempH
                    prob2 = tempProb
                    count = 1
            else:
                if (tempX > ((x2 / count) - accuracy)) and (tempX < ((x2 / count) + accuracy)) and (
                        tempY > ((y2 / count) - accuracy)) and (tempY < ((y2 / count) + accuracy)):
                    x2 = x2 + tempX
                    y2 = y2 + tempY
                    w2 = w2 + tempW
                    h2 = h2 + tempH
                    prob2 = prob2 + tempProb
                    count = count + 1
                else:  # for next object of the same class
                    results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [count,
                                                                                                     (prob2 / count),
                                                                                                     cl, (x2 / count),
                                                                                                     (y2 / count),
                                                                                                     (w2 / count),
                                                                                                     (h2 / count)]
                    x2 = tempX
                    y2 = tempY
                    w2 = tempW
                    h2 = tempH
                    prob2 = tempProb
                    count = 1
                results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [count, (prob2 / count),
                                                                                                 cl, (x2 / count),
                                                                                                 (y2 / count),
                                                                                                 (w2 / count),
                                                                                                 (h2 / count)]
                if i < bigTable.shape[0]:
                    reset = 1
        if i == (bigTable.shape[0] - 1):
            results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [count, (prob2 / count), cl,
                                                                                             (x2 / count), (y2 / count),
                                                                                             (w2 / count), (h2 / count)]
        elif i == 0 and cl != nextCl:
            results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [count, (prob2 / count), cl,
                                                                                             (x2 / count), (y2 / count),
                                                                                             (w2 / count), (h2 / count)]
    xMax = []
    xMin = []
    yMax = []
    yMin = []
    for i in range(results.shape[0]):
        results.loc[i, "x"] = int(results.loc[i, "x"])
        results.loc[i, "y"] = int(results.loc[i, "y"])
        results.loc[i, "w"] = int(results.loc[i, "w"])
        results.loc[i, "h"] = int(results.loc[i, "h"])
        xMax.append(int(results.loc[i, "x"] + (results.loc[i, "w"] / 2)))
        xMin.append(int(results.loc[i, "x"] - (results.loc[i, "w"] / 2)))
        yMax.append(int(results.loc[i, "y"] + (results.loc[i, "h"] / 2)))
        yMin.append(int(results.loc[i, "y"] - (results.loc[i, "h"] / 2)))
    results["xMin"] = xMin
    results["xMax"] = xMax
    results["yMin"] = yMin
    results["yMax"] = yMax
    return results


def drawResults(image, results, labels):
    for i in range(results.shape[0]):
        start_point = (
        int(results.loc[i, "x"] - (results.loc[i, "w"] / 2)), int(results.loc[i, "y"] - (results.loc[i, "h"] / 2)))
        end_point = (
        int(results.loc[i, "x"] + (results.loc[i, "w"] / 2)), int(results.loc[i, "y"] + (results.loc[i, "h"] / 2)))
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = cv2.rectangle(image, start_point, end_point, colour, 1)
        for j in range(labels.shape[0]):
            c = labels.loc[j, "class"]
            if c == results.loc[i, "class"]:
                name = labels.loc[j, "name"]
                break
        cv2.putText(image, name, (
        int(results.loc[i, "x"] - (results.loc[i, "w"] / 2)), int(results.loc[i, "y"] - (results.loc[i, "h"] / 2) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    cv2.imshow("final predictions", image)
    cv2.waitKey(0)


def readGroundTruths(gtLoc, imgID, width, height):
    bboxes = pd.read_csv(gtLoc + "labels/detections.csv")
    gtLabels = pd.read_csv(gtLoc + "metadata/classes.csv")
    gt = pd.DataFrame(columns=bboxes.columns)
    gt = gt.append(bboxes.loc[bboxes["ImageID"] == imgID], ignore_index=True)
    del gt["Source"]
    del gt["Confidence"]
    del gt["IsOccluded"]
    del gt["IsTruncated"]
    del gt["IsGroupOf"]
    del gt["IsDepiction"]
    del gt["IsInside"]
    for i in range(gt.shape[0]):
        l = gt.loc[i, "LabelName"]
        for j in range(gtLabels.shape[0]):
            if gtLabels.loc[j, "ID"] == l:
                gt.loc[i, "LabelName"] = gtLabels.loc[j, "label"].lower()
                break
    for i in range(gt.shape[0]):
        gt.loc[i, "XMin"] = int(gt.loc[i, "XMin"] * width)
        gt.loc[i, "XMax"] = int(gt.loc[i, "XMax"] * width)
        gt.loc[i, "YMin"] = int(gt.loc[i, "YMin"] * height)
        gt.loc[i, "YMax"] = int(gt.loc[i, "YMax"] * height)
    x = []
    y = []
    w = []
    h = []
    for i in range(gt.shape[0]):
        w.append(gt.loc[i, "XMax"] - gt.loc[i, "XMin"])
        h.append(gt.loc[i, "YMax"] - gt.loc[i, "YMin"])
        x.append(int(gt.loc[i, "XMax"] - (w[i] / 2)))
        y.append(int(gt.loc[i, "YMax"] - (h[i] / 2)))
    gt["x"] = x
    gt["y"] = y
    gt["w"] = w
    gt["h"] = h
    return gt


# searches for the first good match
def evaluateIOU(results, groundTruths, iouAccuracy):
    iou = []
    for i in range(results.shape[0]):
        extension = 0  # reduction of accuracy if corresponding ground truth is not found in first attempt
        ratio = 0
        for k in range(5):
            for j in range(groundTruths.shape[0]):
                if groundTruths.loc[j, "LabelName"] == results.loc[i, "labelName"] and (math.fabs(groundTruths.loc[j, "x"] - results.loc[i, "x"]) < (iouAccuracy + extension)) and (math.fabs(groundTruths.loc[j, "y"] - results.loc[i, "y"]) < (iouAccuracy + extension)) and (math.fabs(groundTruths.loc[j, "w"] - results.loc[i, "w"]) < (iouAccuracy*3 + extension)) and (math.fabs(groundTruths.loc[j, "h"] - results.loc[i, "h"]) < (iouAccuracy*3 + extension)):
                    xRight, xLeft, yBottom, yTop = getCorners(groundTruths.loc[j, "XMax"], results.loc[i, "xMax"],
                                                              groundTruths.loc[j, "XMin"], results.loc[i, "xMin"],
                                                              groundTruths.loc[j, "YMax"], results.loc[i, "yMax"],
                                                              groundTruths.loc[j, "YMin"], results.loc[i, "yMin"])
                    intersection = (xRight - xLeft) * (yBottom - yTop)
                    union = (results.loc[i, "w"] * results.loc[i, "h"]) + (groundTruths.loc[j, "w"] * groundTruths.loc[j, "h"]) - intersection
                    # print(xRight, xLeft, yBottom, yTop)
                    # print(intersection, union)
                    if (intersection / union) > 0 and (xRight - xLeft) >= 0 and (yBottom - yTop) >= 0:
                        iou.append(intersection / union)
                        ratio = intersection / union
                        break
                    else:
                        ratio = 0
                    # print(iouAccuracy + extension)
            if ratio > 0:
                break
            else:
                extension += 10
        if ratio <= 0:
            iou.append(0)
    # print(iou)
    results["IOU"] = iou


# searches for the best possible match, takes more time
def evaluateIOUv2(results, groundTruths, iouAccuracy):
    iou = []
    for i in range(results.shape[0]):
        extension = 0  # reduction of accuracy if corresponding ground truth is not found in first attempt
        ratio = 0
        previousIOU = 0
        for k in range(5):
            for j in range(groundTruths.shape[0]):
                if groundTruths.loc[j, "LabelName"] == results.loc[i, "labelName"] and (math.fabs(groundTruths.loc[j, "x"] - results.loc[i, "x"]) < (iouAccuracy + extension)) and (math.fabs(groundTruths.loc[j, "y"] - results.loc[i, "y"]) < (iouAccuracy + extension)) and (math.fabs(groundTruths.loc[j, "w"] - results.loc[i, "w"]) < (iouAccuracy*3 + extension)) and (math.fabs(groundTruths.loc[j, "h"] - results.loc[i, "h"]) < (iouAccuracy*3 + extension)):
                    xRight, xLeft, yBottom, yTop = getCorners(groundTruths.loc[j, "XMax"], results.loc[i, "xMax"],
                                                              groundTruths.loc[j, "XMin"], results.loc[i, "xMin"],
                                                              groundTruths.loc[j, "YMax"], results.loc[i, "yMax"],
                                                              groundTruths.loc[j, "YMin"], results.loc[i, "yMin"])
                    intersection = (xRight - xLeft) * (yBottom - yTop)
                    union = (results.loc[i, "w"] * results.loc[i, "h"]) + (groundTruths.loc[j, "w"] * groundTruths.loc[j, "h"]) - intersection
                    # print(xRight, xLeft, yBottom, yTop)
                    # print(intersection, union)
                    if (intersection / union) > 0 and (xRight - xLeft) >= 0 and (yBottom - yTop) >= 0:
                        if (intersection / union) >= previousIOU:
                            previousIOU = intersection / union
                            ratio = previousIOU
                        else:
                            ratio = 0
                    else:
                        ratio = 0
                    # print(iouAccuracy + extension)
            if ratio <= 0:
                extension += 10
        iou.append(previousIOU)
    # print(iou)
    results["IOU"] = iou


# chooses corners of the intersection rectangle
def getCorners(gtXmax, rXmax, gtXmin, rXmin, gtYmax, rYmax, gtYmin, rYmin):
    if gtXmax < rXmax:
        xRight = gtXmax
    else:
        xRight = rXmax
    if gtXmin > rXmin:
        xLeft = gtXmin
    else:
        xLeft = rXmin
    if gtYmax < rYmax:
        yBottom = gtYmax
    else:
        yBottom = rYmax
    if gtYmin > rYmin:
        yTop = gtYmin
    else:
        yTop = rYmin
    return xRight, xLeft, yBottom, yTop


# adds border values to bboxes
def modifyOriginalData(vanillaYOLO):
    vanillaYOLO.sort_values(["class", "probability"], ascending=[True, False], inplace=True)
    vanillaYOLO.reset_index(drop=True, inplace=True)
    xMax = []
    xMin = []
    yMax = []
    yMin = []
    for i in range(vanillaYOLO.shape[0]):
        vanillaYOLO.loc[i, "x"] = int(vanillaYOLO.loc[i, "x"])
        vanillaYOLO.loc[i, "y"] = int(vanillaYOLO.loc[i, "y"])
        vanillaYOLO.loc[i, "w"] = int(vanillaYOLO.loc[i, "w"])
        vanillaYOLO.loc[i, "h"] = int(vanillaYOLO.loc[i, "h"])
        xMax.append(int(vanillaYOLO.loc[i, "x"] + (vanillaYOLO.loc[i, "w"] / 2)))
        xMin.append(int(vanillaYOLO.loc[i, "x"] - (vanillaYOLO.loc[i, "w"] / 2)))
        yMax.append(int(vanillaYOLO.loc[i, "y"] + (vanillaYOLO.loc[i, "h"] / 2)))
        yMin.append(int(vanillaYOLO.loc[i, "y"] - (vanillaYOLO.loc[i, "h"] / 2)))
    vanillaYOLO["xMin"] = xMin
    vanillaYOLO["xMax"] = xMax
    vanillaYOLO["yMin"] = yMin
    vanillaYOLO["yMax"] = yMax


def addLabelNames(table, labels):
    labelName = []
    for i in range(table.shape[0]):
        for j in range(labels.shape[0]):
            if table.loc[i, "class"] == labels.loc[j, "class"]:
                labelName.append(labels.loc[j, "name"])
                break
    table["labelName"] = labelName


# adds weights to degraded results from rotated image
def modifyWeights(org, rot, accuracy):
    # print(org)
    # print(rot)
    weights = []
    if rot.shape[0] <= org.shape[0]:
        for i in range(rot.shape[0]):
            for j in range(org.shape[0]):
                if rot.loc[i, "class"] == org.loc[j, "class"] and (math.fabs(rot.loc[i, "x"] - org.loc[j, "x"]) < accuracy) and (math.fabs(rot.loc[i, "y"] - org.loc[j, "y"]) < accuracy):
                    if rot.loc[i, "IOU"] > org.loc[j, "IOU"]:
                        w = org.loc[j, "probability"] / rot.loc[i, "probability"]
                        # w = w * 1.2   # strengthens the weight
                        rot.loc[i, "probability"] = rot.loc[i, "probability"] * w
                        # print(rot.loc[i, "probability"])
                        weights.append(w)
                    else:
                        weights.append(1)
                    break
            if len(weights) < i + 1:
                weights.append(1)
    else:
        for i in range(rot.shape[0]):
            for j in range(org.shape[0]):
                if rot.loc[i, "class"] == org.loc[j, "class"] and (math.fabs(rot.loc[i, "x"] - org.loc[j, "x"]) < accuracy) and (math.fabs(rot.loc[i, "y"] - org.loc[j, "y"]) < accuracy):
                    if rot.loc[i, "IOU"] > org.loc[j, "IOU"]:
                        w = org.loc[j, "probability"] / rot.loc[i, "probability"]
                        # w = w * 1.2   # strengthens the weight
                        rot.loc[i, "probability"] = rot.loc[i, "probability"] * w
                        # print(rot.loc[i, "probability"])
                        weights.append(w)
                    else:
                        weights.append(1)
                    break
            if len(weights) < i + 1:
                weights.append(1)
    # print(weights)
    rot["weight"] = weights


# adds custom weight value to all results from the image
def modifyAllWeights(table):
    w = 1.2
    weights = []
    for i in range(table.shape[0]):
        table.loc[i, "probability"] = table.loc[i, "probability"] * w
        weights.append(w)
    table["weight"] = weights


# adds weights equal 1 to results from the image
def addWeights(table):
    weights = []
    for i in range(table.shape[0]):
        weights.append(1)
    table["weight"] = weights


# uses weighted average
def calculateResultsV2(bigTable, accuracy):
    results = pd.DataFrame(columns=["hits", "probability", "class", "x", "y", "w", "h"])
    if bigTable.empty == True:
        return results
    x2 = 0
    y2 = 0
    w2 = 0
    h2 = 0
    prob2 = 0
    count = bigTable.loc[0, "weight"]
    hits = 1
    cl = -1
    reset = 0
    for i in range(bigTable.shape[0]):
        cl = bigTable.loc[i, "class"]
        tempX = bigTable.loc[i, "x"]
        tempY = bigTable.loc[i, "y"]
        tempW = bigTable.loc[i, "w"]
        tempH = bigTable.loc[i, "h"]
        tempProb = bigTable.loc[i, "probability"]
        if i < (bigTable.shape[0] - 1):
            nextCl = bigTable.loc[i + 1, "class"]
        if i == 0 or reset == 1:
            x2 = tempX
            y2 = tempY
            w2 = tempW
            h2 = tempH
            prob2 = tempProb
            count = bigTable.loc[i, "weight"]
            reset = 0
        else:
            if cl == nextCl:
                if (tempX > ((x2 / count) - accuracy)) and (tempX < ((x2 / count) + accuracy)) and (
                        tempY > ((y2 / count) - accuracy)) and (tempY < ((y2 / count) + accuracy)):
                    x2 = x2 + tempX
                    y2 = y2 + tempY
                    w2 = w2 + tempW
                    h2 = h2 + tempH
                    prob2 = prob2 + tempProb
                    count = count + bigTable.loc[i, "weight"]
                    hits = hits + 1
                else:  # for next object of the same class
                    results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [hits,
                                                                                                     (prob2 / count),
                                                                                                     cl, (x2 / count),
                                                                                                     (y2 / count),
                                                                                                     (w2 / count),
                                                                                                     (h2 / count)]
                    x2 = tempX
                    y2 = tempY
                    w2 = tempW
                    h2 = tempH
                    prob2 = tempProb
                    count = bigTable.loc[i, "weight"]
                    hits = 1
            else:
                if (tempX > ((x2 / count) - accuracy)) and (tempX < ((x2 / count) + accuracy)) and (
                        tempY > ((y2 / count) - accuracy)) and (tempY < ((y2 / count) + accuracy)):
                    x2 = x2 + tempX
                    y2 = y2 + tempY
                    w2 = w2 + tempW
                    h2 = h2 + tempH
                    prob2 = prob2 + tempProb
                    count = count + bigTable.loc[i, "weight"]
                    hits = hits + 1
                else:  # for next object of the same class
                    results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [hits,
                                                                                                     (prob2 / count),
                                                                                                     cl, (x2 / count),
                                                                                                     (y2 / count),
                                                                                                     (w2 / count),
                                                                                                     (h2 / count)]
                    x2 = tempX
                    y2 = tempY
                    w2 = tempW
                    h2 = tempH
                    prob2 = tempProb
                    count = bigTable.loc[i, "weight"]
                    hits = 1
                results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [hits, (prob2 / count),
                                                                                                 cl, (x2 / count),
                                                                                                 (y2 / count),
                                                                                                 (w2 / count),
                                                                                                 (h2 / count)]
                if i < bigTable.shape[0]:
                    reset = 1
        if i == (bigTable.shape[0] - 1):
            results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [hits, (prob2 / count), cl,
                                                                                             (x2 / count), (y2 / count),
                                                                                             (w2 / count), (h2 / count)]
        elif i == 0 and cl != nextCl:
            results.loc[0 if pd.isnull(results.index.max()) else results.index.max() + 1] = [hits, (prob2 / count), cl,
                                                                                             (x2 / count), (y2 / count),
                                                                                             (w2 / count), (h2 / count)]
    xMax = []
    xMin = []
    yMax = []
    yMin = []
    for i in range(results.shape[0]):
        results.loc[i, "x"] = int(results.loc[i, "x"])
        results.loc[i, "y"] = int(results.loc[i, "y"])
        results.loc[i, "w"] = int(results.loc[i, "w"])
        results.loc[i, "h"] = int(results.loc[i, "h"])
        xMax.append(int(results.loc[i, "x"] + (results.loc[i, "w"] / 2)))
        xMin.append(int(results.loc[i, "x"] - (results.loc[i, "w"] / 2)))
        yMax.append(int(results.loc[i, "y"] + (results.loc[i, "h"] / 2)))
        yMin.append(int(results.loc[i, "y"] - (results.loc[i, "h"] / 2)))
    results["xMin"] = xMin
    results["xMax"] = xMax
    results["yMin"] = yMin
    results["yMax"] = yMax
    return results


if __name__ == '__main__':
    # used for testing on a single picture
    # location = "C:/images"
    # imgName = "00e8e1122ed9f31c.jpg"
    # nr = "165"

    location = sys.argv[1]
    imgName = sys.argv[2]
    nr = sys.argv[3]
    # print(sys.argv[1])
    # print(sys.argv[2])
    # print(sys.argv[3])

    mode = 3    # 0 - 3 pictures,   1 - original & mirrored,   2 - original & rotated,  3 - 3 pictures & 3 versions of calculating results
    saveFiles = 1
    accuracy = 100   # in pixels
    # accuracy = int(width / 10)    # relative to image size
    probThresh = 0.5
    iouAccuracy = 100   # in pixels
    # iouAccuracy = int(width * 20 / 100)   # relative to image size
    if mode == 0 or mode == 3:
        image, rImage, original, mirror, rotated = read3images(location, imgName, nr)
    elif mode == 1:
        image, original, mirror = read2imagesM(location, imgName, nr)
    elif mode == 2:
        image, rImage, original, rotated = read2imagesR(location, imgName, nr)

    height, width, channels = image.shape
    centreX = width / 2
    centreY = height / 2
    if mode == 0 or mode == 2 or mode == 3:
        rHeight, rWidth, rChannels = rImage.shape
        rCentreX = rWidth / 2
        rCentreY = rHeight / 2
        marginW = rWidth - width
        marginH = rHeight - height
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(height, width)
    # print(rHeight, rWidth)
    # print(original)
    # print(mirror)
    # print(rotated)
    transformOrgCoordinates(original, width, height)
    if mode == 0 or mode == 1 or mode == 3:
        transformMirrCoordinates(mirror, width, height)
    if mode == 0 or mode == 2 or mode == 3:
        transformRotCoordinates(rotated, rWidth, rHeight, marginW, marginH)

    labelsLocation = "C:/Yolo2017/darknet-master/build/darknet/x64/names.csv"
    labels = pd.read_csv(labelsLocation)
    # print(labels)
    #---------------------------------------------------------------------------------------------
    if mode == 3:
        original2 = original.copy(deep=True)
        removeUnclassified(original2)
        removeLowProbabilities(original2, probThresh)
        original2.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        original2.reset_index(drop=True, inplace=True)
        original2 = calculateResults(original2, accuracy)

        rotated2 = rotated.copy(deep=True)
        removeUnclassified(rotated2)
        removeLowProbabilities(rotated2, probThresh)
        rotated2.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        rotated2.reset_index(drop=True, inplace=True)
        rotated2 = calculateResults(rotated2, accuracy)

        mirror2 = mirror.copy(deep=True)
        removeUnclassified(mirror2)
        removeLowProbabilities(mirror2, probThresh)
        mirror2.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        mirror2.reset_index(drop=True, inplace=True)
        mirror2 = calculateResults(mirror2, accuracy)
        # print(original2)
        # print(mirror2)
        # print(rotated2)
    # ---------------------------------------------------------------------------------------------
    results = pd.DataFrame(columns=["hits", "probability", "class", "x", "y", "w", "h"])
    if mode == 0 or mode == 3:
        bigTable = pd.concat([original, mirror, rotated])
    elif mode == 1:
        bigTable = pd.concat([original, mirror])
    elif mode == 2:
        bigTable = pd.concat([original, rotated])
    # print(bigTable)
    bigTable.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
    bigTable.reset_index(drop=True, inplace=True)
    # print(bigTable)

    # removing hits without classification
    removeUnclassified(bigTable)
    # print(bigTable)

    # removing low probability hits
    removeLowProbabilities(bigTable, probThresh)
    # print(bigTable)

    results = calculateResults(bigTable, accuracy)
    # print(results)

    gtLoc = "C:/Users/ja/fiftyone/open-images-v6/validation/"
    imgNameShort = imgName[:-4]
    groundTruths = readGroundTruths(gtLoc, imgNameShort, width, height)
    groundTruths.sort_values(["x", "y"], ascending=[True, True], inplace=True)
    groundTruths.reset_index(drop=True, inplace=True)
    # print(groundTruths)
    if saveFiles == 1:
        groundTruths.to_csv("ground_truths.csv")

    addLabelNames(results, labels)
    evaluateIOUv2(results, groundTruths, iouAccuracy)
    print("enhanced results:\n", results, "\n")
    if saveFiles == 1:
        results.to_csv("final_resultsV1.csv")

    vanillaYOLO = pd.read_csv(location + "/" + nr + "/data/vanillaYOLOresults.csv")
    # removing low probability samples from unmodified YOLO results is optional
    # removeLowProbabilities(vanillaYOLO, probThresh)

    transformOrgCoordinates(vanillaYOLO, width, height)
    modifyOriginalData(vanillaYOLO)
    addLabelNames(vanillaYOLO, labels)
    evaluateIOUv2(vanillaYOLO, groundTruths, iouAccuracy)
    print("results after basic YOLO:\n", vanillaYOLO, "\n")
    if saveFiles == 1:
        vanillaYOLO.to_csv("YOLO.csv")

    # ---------------------------------------------------------------------------------------------
    if mode == 3:
        addLabelNames(original2, labels)
        evaluateIOUv2(original2, groundTruths, iouAccuracy)
        original2.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        original2.reset_index(drop=True, inplace=True)

        addLabelNames(rotated2, labels)
        evaluateIOUv2(rotated2, groundTruths, iouAccuracy)
        rotated2.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        rotated2.reset_index(drop=True, inplace=True)
        rotated3 = rotated2.copy(deep=True)
        # print(rotated3)

        wAccuracy = 50
        modifyWeights(original2, rotated2, wAccuracy)
        # modifyAllWeights(rotated2)
        addWeights(original2)

        addLabelNames(mirror2, labels)
        evaluateIOUv2(mirror2, groundTruths, iouAccuracy)
        addWeights(mirror2)
        mirror2.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        mirror2.reset_index(drop=True, inplace=True)

        # print(original2)
        # print(mirror2)
        # print(rotated2)
        # print(rotated3)
        bigTable2 = pd.concat([original2, mirror2, rotated2])
        bigTable2.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        bigTable2.reset_index(drop=True, inplace=True)
        # print("\n", bigTable2)
        results2 = pd.DataFrame(columns=["hits", "probability", "class", "x", "y", "w", "h"])
        results2 = calculateResultsV2(bigTable2, accuracy)
        addLabelNames(results2, labels)
        evaluateIOUv2(results2, groundTruths, iouAccuracy)
        print("enhanced results v2:\n", results2)
        if saveFiles == 1:
            results2.to_csv("final_resultsV2.csv")

        results3 = pd.DataFrame(columns=["hits", "probability", "class", "x", "y", "w", "h"])
        bigTable3 = pd.concat([original2, mirror2, rotated3])
        bigTable3.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
        bigTable3.reset_index(drop=True, inplace=True)
        # print("\n", bigTable3)
        results3 = calculateResults(bigTable3, accuracy)
        addLabelNames(results3, labels)
        evaluateIOUv2(results3, groundTruths, iouAccuracy)
        print("enhanced results v3:\n", results3)
        if saveFiles == 1:
            results3.to_csv("final_resultsV3.csv")
    # ---------------------------------------------------------------------------------------------

    drawResults(image, results, labels)
    if saveFiles == 1:
        cv2.imwrite("final.jpg", image)
    cv2.destroyAllWindows()
