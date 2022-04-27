import sys
sys.path.append("C:/Users/ja/PycharmProjects/resultsAggregation/venv/Lib/site-packages/")
import pandas as pd


def removeLowProbabilities(table, probThresh):
    for i in range(table.shape[0]):
        if table.loc[i, "probability"] < probThresh:
            table.drop([i], inplace=True)
    table.reset_index(drop=True, inplace=True)


def sortTable(table):
    table.sort_values(["class", "x", "y"], ascending=[True, True, True], inplace=True)
    table.reset_index(drop=True, inplace=True)


def removeNoGTSamples(table):
    for i in range(table.shape[0]):
        if table.loc[i, "IOU"] == 0:
            table.drop([i], inplace=True)
    table.reset_index(drop=True, inplace=True)


if __name__ == '__main__':
    location = sys.argv[1]
    filename = sys.argv[2]
    mode = sys.argv[3]

    # used for testing
    # location = "C:/images"
    # filename = "final_resultsV1.csv"
    # mode = "1"

    images = 0
    totalSummary = pd.DataFrame(columns=["probability", "IOU", "difference"])
    threshold = 0.5
    saveFiles = 1
    for i in range(1000):
        print("----- image", i, "-----")
        prob = []
        iou = []
        difference = []
        summary = pd.DataFrame(columns=["probability", "IOU", "difference"])
        enhancedResults = pd.read_csv(location + "/" + str(i) + "/data/" + filename)
        yoloResults = pd.read_csv(location + "/" + str(i) + "/data/YOLO.csv")
        if enhancedResults.empty == True or yoloResults.empty == True:
            continue
        images += 1
        removeLowProbabilities(yoloResults, threshold)
        sortTable(enhancedResults)
        sortTable(yoloResults)
        removeNoGTSamples(enhancedResults)
        removeNoGTSamples(yoloResults)
        print(enhancedResults)
        print(enhancedResults.shape[0], "rows\n")
        print(yoloResults)
        print(yoloResults.shape[0], "rows\n")

        k = 0
        if yoloResults.shape[0] >= enhancedResults.shape[0]:
            for j in range(enhancedResults.shape[0]):
                while enhancedResults.loc[j, "class"] != yoloResults.loc[k, "class"] and k < yoloResults.shape[0] - 1:
                    k += 1
                if enhancedResults.loc[j, "class"] != yoloResults.loc[k, "class"]:
                    break
                probSample = enhancedResults.loc[j, "probability"] - yoloResults.loc[k, "probability"]
                iouSample = enhancedResults.loc[j, "IOU"] - yoloResults.loc[k, "IOU"]
                prob.append(probSample)
                iou.append(iouSample)
                if probSample >= 0 and iouSample >= 0:
                    difference.append("positive")
                elif probSample < 0 and iouSample >= 0:
                    difference.append("expected trade-off")
                elif probSample < 0 and iouSample < 0:
                    difference.append("negative")
                else:
                    difference.append("opposite trade-off")
        else:
            for j in range(yoloResults.shape[0]):
                while enhancedResults.loc[k, "class"] != yoloResults.loc[j, "class"] and k < enhancedResults.shape[0] - 1:
                    k += 1
                if enhancedResults.loc[k, "class"] != yoloResults.loc[j, "class"]:
                    break
                probSample = enhancedResults.loc[k, "probability"] - yoloResults.loc[j, "probability"]
                iouSample = enhancedResults.loc[k, "IOU"] - yoloResults.loc[j, "IOU"]
                prob.append(probSample)
                iou.append(iouSample)
                if probSample >= 0 and iouSample >= 0:
                    difference.append("positive")
                elif probSample < 0 and iouSample >= 0:
                    difference.append("expected trade-off")
                elif probSample < 0 and iouSample < 0:
                    difference.append("negative")
                else:
                    difference.append("opposite trade-off")
        summary["probability"] = prob
        summary["IOU"] = iou
        summary["difference"] = difference
        dfs = [totalSummary, summary]
        totalSummary = pd.concat(dfs)

        print(summary)
        print("\nmean probability changes: ", summary["probability"].mean())
        print("mean IoU changes: ", summary["IOU"].mean())
        print("types of differences:\n", summary["difference"].value_counts())
        if saveFiles == 1:
            summary.to_csv(location + "/" + str(i) + "/data/" + str("summary" + mode + ".csv"))
        summary = pd.DataFrame(columns=summary.columns)
        print("\n")

    print("\n------------------------------------------\n")
    if totalSummary.empty == False:
        totalSummary.reset_index(drop=True, inplace=True)
        print(totalSummary)
        print("\nTypes of differences from all images:\n", totalSummary["difference"].value_counts())

        print("\nProbability changes from all images")
        print("mean: ", totalSummary["probability"].mean())
        print("variance", totalSummary["probability"].var())
        print("standard deviation", totalSummary["probability"].std())
        print("max", totalSummary["probability"].max())
        print("min", totalSummary["probability"].min())
        print("median", totalSummary["probability"].median())

        print("\nIoU changes from all images")
        print("mean", totalSummary["IOU"].mean())
        print("variance", totalSummary["IOU"].var())
        print("standard deviation", totalSummary["IOU"].std())
        print("max", totalSummary["IOU"].max())
        print("min", totalSummary["IOU"].min())
        print("median", totalSummary["IOU"].median())
        if saveFiles == 1 or saveFiles == 2:
            totalSummary.to_csv("C:/images/totalSummary" + mode + ".csv")
            statistics = open(r"C:/images/statistics" + mode + ".txt", "w")
            statistics.write("Types of differences from all images:\n" + str(totalSummary["difference"].value_counts()) + "\n")

            statistics.write("\nProbability changes from all images\n")
            statistics.write("mean: " + str(totalSummary["probability"].mean()) + "\n")
            statistics.write("max: " + str(totalSummary["probability"].max()) + "\n")
            statistics.write("min: " + str(totalSummary["probability"].min()) + "\n")
            statistics.write("median: " + str(totalSummary["probability"].median()) + "\n")
            statistics.write("variance: " + str(totalSummary["probability"].var()) + "\n")
            statistics.write("standard deviation: " + str(totalSummary["probability"].std()) + "\n")

            statistics.write("\nIoU changes from all images\n")
            statistics.write("mean: " + str(totalSummary["IOU"].mean()) + "\n")
            statistics.write("max: " + str(totalSummary["IOU"].max()) + "\n")
            statistics.write("min: " + str(totalSummary["IOU"].min()) + "\n")
            statistics.write("median: " + str(totalSummary["IOU"].median()) + "\n")
            statistics.write("variance: " + str(totalSummary["IOU"].var()) + "\n")
            statistics.write("standard deviation: " + str(totalSummary["IOU"].std()) + "\n")

            statistics.write("number of recognized objects with ground truths: " + str(totalSummary.shape[0]) + "\n")
            statistics.write("number of correct images taken for statistics calculation:" + str(images))
            statistics.close()
    print("\nnumber of recognized objects with ground truths:", totalSummary.shape[0])
    print("number of correct images taken for statistics calculation:", images)
