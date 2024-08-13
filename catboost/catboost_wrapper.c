#include "catboost_wrapper.h"

static TypeGetErrorString GetErrorStringFn = NULL;
static TypeModelCalcerCreate ModelCalcerCreateFn = NULL;
static TypeLoadFullModelFromBuffer LoadFullModelFromBufferFn = NULL;
static TypeCalcModelPredictionSingle CalcModelPredictionSingleFn = NULL;
static TypeCalcModelPrediction CalcModelPredictionFn = NULL;
static TypeGetFloatFeaturesCount GetFloatFeaturesCountFn = NULL;
static TypeGetCatFeaturesCount GetCatFeaturesCountFn = NULL;
static TypeGetDimensionsCount GetDimensionsCountFn = NULL;
static TypeSetPredictionTypeString SetPredictionTypeStringFn = NULL;
static TypeGetModelUsedFeaturesNames GetModelUsedFeaturesNamesFn = NULL;
static TypeGetModelInfoValue GetModelInfoValueFn = NULL;

const char* WrapGetErrorString() {
	return GetErrorStringFn();
}

ModelCalcerHandle* WrapModelCalcerCreate() {
	return ModelCalcerCreateFn();
}

bool WrapLoadFullModelFromBuffer(ModelCalcerHandle* modelHandle, const void* binaryBuffer, size_t binaryBufferSize) {
	return LoadFullModelFromBufferFn(modelHandle, binaryBuffer, binaryBufferSize);
}

bool WrapCalcModelPredictionSingle(
	ModelCalcerHandle* modelHandle, 
	const float* floatFeatures, 
	size_t floatFeaturesSize, 
	const char** catFeatures, 
	size_t catFeaturesSize, 
	double* result, size_t resultSize) {
	return CalcModelPredictionSingleFn(modelHandle, floatFeatures, floatFeaturesSize, catFeatures, catFeaturesSize, result, resultSize);
}

bool WrapCalcModelPrediction(
    ModelCalcerHandle* modelHandle,
    size_t docCount,
    const float** floatFeatures, size_t floatFeaturesSize,
    const char*** catFeatures, size_t catFeaturesSize,
    double* result, size_t resultSize) {
		return CalcModelPredictionFn(modelHandle, docCount, floatFeatures, floatFeaturesSize, catFeatures, catFeaturesSize, result, resultSize);
}

bool WrapGetModelUsedFeaturesNames(ModelCalcerHandle* modelHandle, char*** featureNames, size_t* featureCount) {
	return GetModelUsedFeaturesNamesFn(modelHandle, featureNames, featureCount);
}

size_t WrapGetFloatFeaturesCount(ModelCalcerHandle* modelHandle) {
	return GetFloatFeaturesCountFn(modelHandle);
}

size_t WrapGetCatFeaturesCount(ModelCalcerHandle* modelHandle) {
	return GetCatFeaturesCountFn(modelHandle);
}

size_t WrapGetDimensionsCount(ModelCalcerHandle* modelHandle) {
	return GetDimensionsCountFn(modelHandle);
}

bool WrapSetPredictionTypeString(ModelCalcerHandle* modelHandle, const char* predictionTypeStr) {
	return SetPredictionTypeStringFn(modelHandle, predictionTypeStr);
}

const char* WrapGetModelInfoValue(ModelCalcerHandle* modelHandle, const char* keyPtr, size_t keySize) {
	return GetModelInfoValueFn(modelHandle, keyPtr, keySize);
}

void SetCalcModelPredictionSingleFn(void *fn) {
	CalcModelPredictionSingleFn = ((TypeCalcModelPredictionSingle) fn);
}

void SetModelCalcerCreateFn(void *fn) {
	ModelCalcerCreateFn = ((TypeModelCalcerCreate) fn);
}

void SetLoadFullModelFromBufferFn(void *fn) {
	LoadFullModelFromBufferFn = ((TypeLoadFullModelFromBuffer) fn);
}

void SetGetErrorStringFn(void *fn) {
	GetErrorStringFn = ((TypeGetErrorString) fn);
}

void SetCalcModelPredictionFn(void *fn) {
	CalcModelPredictionFn = ((TypeCalcModelPrediction) fn);
}

void SetGetFloatFeaturesCountFn(void *fn) {
	GetFloatFeaturesCountFn = ((TypeGetFloatFeaturesCount) fn);
}

void SetGetCatFeaturesCountFn(void *fn) {
	GetCatFeaturesCountFn = ((TypeGetCatFeaturesCount) fn);
}

void SetGetDimensionsCountFn(void *fn) {
	GetDimensionsCountFn = ((TypeGetDimensionsCount) fn);
}

void SetSetPredictionTypeStringFn(void *fn) {
	SetPredictionTypeStringFn = ((TypeSetPredictionTypeString) fn);
}

void SetGetModelUsedFeaturesNamesFn(void *fn) {
	GetModelUsedFeaturesNamesFn = ((TypeGetModelUsedFeaturesNames) fn);
}

void SetGetModelInfoValueFn(void *fn) {
	GetModelInfoValueFn = ((TypeGetModelInfoValue) fn);
}

char*** makeCharArray2D(int size) {
	return calloc(sizeof(char**), size);
}

char** makeCharArray1D(int size) {
    return calloc(sizeof(char*), size);
}

void freeCharArray1D(char **array, int size) {
	int i;
	for (i = 0; i < size; i++)
		free(array[i]);
	free(array);
}

void freeCharArray2D(char ***a, int sizeX, int sizeY) {
	int i;
	for (i = 0; i < sizeX; i++)
		freeCharArray1D(a[i], sizeY);
	free(a);
}

void setCharArray1D(char **array, char *s, int n) {
    array[n] = s;
}

 void setCharArray2D(char ***a, char **s, int n) {
	a[n] = s;
}

float** makeFloatArray2D(int size) {
	return calloc(sizeof(float*), size);
}

void setFloatArray2D(float **a, float *f, int n) {
	a[n] = f;
}

