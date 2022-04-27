#ifndef __PREPROCIMG_H
#define __PREPROCIMG_H

#ifdef __cplusplus
extern "C" void mirrorImg(char *path);
extern "C" void rotatedImg(char *path);
#else
void mirrorImg(char *path);
void rotatedImg(char *path);
#endif

#endif