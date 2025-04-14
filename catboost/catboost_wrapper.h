#include <stdio.h>
#include <stdlib.h>
#include "c_api.h"

void SetGetErrorStringFn(void *fn);
void SetCalcModelPredictionSingleFn(void *fn);
void SetModelCalcerCreateFn(void *fn);
void SetModelCalcerDeleteFn(void *fn);
void SetLoadFullModelFromBufferFn(void *fn);
void SetCalcModelPredictionFn(void *fn);
void SetGetFloatFeaturesCountFn(void *fn);
void SetGetCatFeaturesCountFn(void *fn);
void SetGetDimensionsCountFn(void *fn);
void SetSetPredictionTypeStringFn(void *fn);
void SetGetModelUsedFeaturesNamesFn(void *fn);
void SetGetModelInfoValueFn(void *fn);
void SetGetCatFeatureIndicesFn(void *fn);
void SetGetFloatFeatureIndicesFn(void *fn);
void SetGetSupportedEvaluatorTypesFn(void *fn);
void SetGetEnableGPUEvaluationFn(void *fn);

const char *WrapGetErrorString();
ModelCalcerHandle *WrapModelCalcerCreate();
void WrapModelCalcerDelete(ModelCalcerHandle *modelHandle);
bool WrapLoadFullModelFromBuffer(ModelCalcerHandle *modelHandle, const void *binaryBuffer, size_t binaryBufferSize);
bool WrapCalcModelPredictionSingle(ModelCalcerHandle *modelHandle, const float *floatFeatures, size_t floatFeaturesSize, const char **catFeatures, size_t catFeaturesSize, double *result, size_t resultSize);
bool WrapCalcModelPrediction(ModelCalcerHandle *modelHandle, size_t docCount, const float **floatFeatures, size_t floatFeaturesSize, const char ***catFeatures, size_t catFeaturesSize, double *result, size_t resultSize);
size_t WrapGetFloatFeaturesCount(ModelCalcerHandle *modelHandle);
size_t WrapGetCatFeaturesCount(ModelCalcerHandle *modelHandle);
size_t WrapGetDimensionsCount(ModelCalcerHandle *modelHandle);
bool WrapSetPredictionTypeString(ModelCalcerHandle *modelHandle, const char *predictionTypeStr);
bool WrapGetModelUsedFeaturesNames(ModelCalcerHandle *modelHandle, char ***featureNames, size_t *featureCount);
const char *WrapGetModelInfoValue(ModelCalcerHandle *modelHandle, const char *keyPtr, size_t keySize);
bool WrapGetCatFeatureIndices(ModelCalcerHandle *modelHandle, size_t **indices, size_t *count);
bool WrapGetFloatFeatureIndices(ModelCalcerHandle *modelHandle, size_t **indices, size_t *count);
bool WrapGetSupportedEvaluatorTypes(ModelCalcerHandle *modelHandle, size_t **formulaEvaluatorTypes, size_t *count);
bool WrapEnableGPUEvaluation(ModelCalcerHandle *modelHandle, int deviceId);

void freeCharArray1D(char **a, int size);
void freeCharArray2D(char ***a, int sizeX, int sizeY);

void setCharArray1D(char **a, char *s, int n);
void setFloatArray2D(float **a, float *f, int n);
void setCharArray2D(char ***a, char **s, int n);

char **makeCharArray1D(int size);
char ***makeCharArray2D(int size);
float **makeFloatArray2D(int size);
