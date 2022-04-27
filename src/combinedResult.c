#include"combinedResult.h"
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"image.h"
#include"blas.h"

void saveResults(partialResults *results, char *name) {
    FILE *f;
    f = fopen(name, "w+");
    /*
    box *bbox;
    float *probabilities;	//prob
    int numberOfProbabilities;	//sum of count2
    int *classes;	//j
    int *hits;	//count or box id
    int numberOfCells;	//sum of count
    int totalNumberOfClasses;	//l.classes
    */
    fprintf(f, "scale,object,cell,class,probability,x,y,w,h\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0;j < results[i].numberOfProbabilities;j++) {
            fprintf(f, "%d,%d,%d,%d,%f,%f,%f,%f,%f\n", i, j, results[i].hits[j], results[i].classes[j], results[i].probabilities[j],
                results[i].bbox[j].x, results[i].bbox[j].y, results[i].bbox[j].w, results[i].bbox[j].h);
        }
    }
    fclose(f);
}