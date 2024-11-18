package catboost

/*
#cgo LDFLAGS: -ldl
#cgo CFLAGS: -O3 -g
#include <dlfcn.h>
#include <catboost_wrapper.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"unsafe"
)

// PredictionType typing inference Model.
type PredictionType string

// EvaluatorType typing device.
type EvaluatorType uint64

const (
	RawFormulaVal       PredictionType = "RawFormulaVal"
	Probablity          PredictionType = "Probability"
	Class               PredictionType = "Class"
	RMSEWithUncertainty PredictionType = "RMSEWithUncertainty"
	Exponent            PredictionType = "Exponent"
)

const (
	// CPU device.
	CPU EvaluatorType = iota
	// GPU device.
	GPU
)

const formatErrorMessage = "%w: %v"

// https://catboost.ai/en/docs/concepts/python-reference_catboost_metadata
const (
	MetaVersionInfo     = "catboost_version_info"
	MetaModelGUID       = "model_guid"
	MetaParams          = "params"
	MetaTrainFinishTime = "train_finish_time"
	MetaTraining        = "training"
	MetaOutputOptions   = "output_options"
)

var (
	ErrLoadFullModelFromFile     = errors.New("failed load model from file")
	ErrLoadFullModelFromBuffer   = errors.New("failed load model from bytes")
	ErrNotFoundLibrary           = errors.New("not found catboost library")
	ErrGetModelUsedFeaturesNames = errors.New("failed get used features name")
	ErrCalcModelPrediction       = errors.New("failed inference model")
	ErrNotSupportedPlatform      = errors.New("supported only Linux and MacOS")
	ErrLoadLibrary               = errors.New("failed loading CatBoost shared library")
	ErrSetPredictionType         = errors.New("failed set prediction type")
	ErrGetIndices                = errors.New("failed get indices")
	ErrGetDevices                = errors.New("failed get list devices")
	ErrEnabledGPU                = errors.New("failed enabled GPU")
	ErrNotSupportedGPU           = errors.New("supported GPU only Linux")
)

var catboostSharedLibraryPath = ""

// Version returns version catboost.
func Version() string {
	return fmt.Sprintf("v%d.%d.%d", C.CATBOOST_APPLIER_MAJOR, C.CATBOOST_APPLIER_MINOR, C.CATBOOST_APPLIER_FIX)
}

// SetSharedLibraryPath set library catboost path.
func SetSharedLibraryPath(path string) {
	catboostSharedLibraryPath = path
}

func initialization() error {
	if !checkPlatform() {
		return ErrNotSupportedPlatform
	}

	if err := initSharedLibraryPath(); err != nil {
		return err
	}

	cName := C.CString(catboostSharedLibraryPath)
	defer C.free(unsafe.Pointer(cName))

	handle := C.dlopen(cName, C.RTLD_LAZY)
	if handle == nil {
		msg := C.GoString(C.dlerror())
		return fmt.Errorf("%w `%s`: %s", ErrLoadLibrary, catboostSharedLibraryPath, msg)
	}

	lib := library{handle}

	// Load function from CatBoost shared library
	lib.RegisterFn("ModelCalcerCreate")
	lib.RegisterFn("LoadFullModelFromBuffer")
	lib.RegisterFn("CalcModelPredictionSingle")
	lib.RegisterFn("CalcModelPrediction")
	lib.RegisterFn("GetErrorString")
	lib.RegisterFn("GetFloatFeaturesCount")
	lib.RegisterFn("GetCatFeaturesCount")
	lib.RegisterFn("GetDimensionsCount")
	lib.RegisterFn("SetPredictionTypeString")
	lib.RegisterFn("GetModelUsedFeaturesNames")
	lib.RegisterFn("GetModelInfoValue")
	lib.RegisterFn("GetCatFeatureIndices")
	lib.RegisterFn("GetFloatFeatureIndices")
	lib.RegisterFn("GetSupportedEvaluatorTypes")
	lib.RegisterFn("EnableGPUEvaluation")

	return nil
}

type library struct {
	handle unsafe.Pointer
}

func (l *library) RegisterFn(fnName string) {
	fnC := getFromLibraryFn(l.handle, fnName)

	switch fnName {
	case "ModelCalcerCreate":
		C.SetModelCalcerCreateFn(fnC)
	case "LoadFullModelFromBuffer":
		C.SetLoadFullModelFromBufferFn(fnC)
	case "CalcModelPredictionSingle":
		C.SetCalcModelPredictionSingleFn(fnC)
	case "CalcModelPrediction":
		C.SetCalcModelPredictionFn(fnC)
	case "GetErrorString":
		C.SetGetErrorStringFn(fnC)
	case "GetFloatFeaturesCount":
		C.SetGetFloatFeaturesCountFn(fnC)
	case "GetCatFeaturesCount":
		C.SetGetCatFeaturesCountFn(fnC)
	case "SetPredictionTypeString":
		C.SetSetPredictionTypeStringFn(fnC)
	case "GetDimensionsCount":
		C.SetGetDimensionsCountFn(fnC)
	case "GetModelUsedFeaturesNames":
		C.SetGetModelUsedFeaturesNamesFn(fnC)
	case "GetModelInfoValue":
		C.SetGetModelInfoValueFn(fnC)
	case "GetCatFeatureIndices":
		C.SetGetCatFeatureIndicesFn(fnC)
	case "GetFloatFeatureIndices":
		C.SetGetFloatFeatureIndicesFn(fnC)
	case "GetSupportedEvaluatorTypes":
		C.SetGetSupportedEvaluatorTypesFn(fnC)
	case "EnableGPUEvaluation":
		C.SetGetEnableGPUEvaluationFn(fnC)
	default:
		panic(fmt.Sprintf("not supported function from catboost library: %s", fnName))
	}
}

func initSharedLibraryPath() error {
	if catboostSharedLibraryPath == "" {
		catboostSharedLibraryPath = os.Getenv("CATBOOST_LIBRARY_PATH")
	}

	if catboostSharedLibraryPath == "" {
		catboostSharedLibraryPath = fmt.Sprintf("/usr/local/lib/libcatboostmodel.%s", getExt())
	}

	if _, err := os.Stat(catboostSharedLibraryPath); errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("%w: %s", ErrNotFoundLibrary, catboostSharedLibraryPath)
	}

	return nil
}

func checkPlatform() bool {
	return slices.Contains([]string{"darwin", "linux"}, runtime.GOOS)
}

func getExt() string {
	ext := "dylib"

	if runtime.GOOS == "linux" {
		ext = "so"
	}

	return ext
}

// LoadFullModelFromFile returns load model from file into given model handle.
func LoadFullModelFromFile(filename string) (*Model, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf(formatErrorMessage, ErrLoadFullModelFromFile, err)
	}

	return LoadFullModelFromBuffer(b)
}

// LoadFullModelFromBuffer returns load model from memory buffer into given model handle.
func LoadFullModelFromBuffer(buffer []byte) (*Model, error) {
	if err := initialization(); err != nil {
		return nil, err
	}

	handler := C.WrapModelCalcerCreate()

	if !C.WrapLoadFullModelFromBuffer(handler, unsafe.Pointer(&buffer[0]), C.size_t(len(buffer))) {
		return nil, fmt.Errorf(formatErrorMessage, ErrLoadFullModelFromBuffer, GetError())
	}

	return &Model{handler: handler, predictionType: RawFormulaVal}, nil
}

// Model is a wrapper over ModelCalcerHandle.
type Model struct {
	handler        unsafe.Pointer
	predictionType PredictionType
}

// GetModelInfoValue returns model metainfo for some key.
// If key is missing in model metainfo storage this method will return "".
func (m *Model) GetModelInfoValue(key string) string {
	keyC := C.CString(key)
	defer C.free(unsafe.Pointer(keyC))

	valueC := C.WrapGetModelInfoValue(m.handler, keyC, C.size_t(len(key)))
	return C.GoString(valueC)
}

// SetPredictionType set prediction type for model evaluation.
// Not use in concurrency mode!!!
// Recommend set prediction type after load model.
func (m *Model) SetPredictionType(p PredictionType) error {
	pC := C.CString(string(p))
	defer C.free(unsafe.Pointer(pC))

	if !C.WrapSetPredictionTypeString(m.handler, pC) {
		return fmt.Errorf("%w `%s`: %s", ErrSetPredictionType, p, GetError().Error())
	}

	m.predictionType = p

	return nil
}

// GetSupportedEvaluatorTypes returns supported formula evaluator types.
func (m *Model) GetSupportedEvaluatorTypes() ([]EvaluatorType, error) {
	devicesNum := uint64(2)

	devicesTmp := make([]*uint64, devicesNum)
	devicesC := (*C.size_t)(devicesTmp[0])
	defer C.free(unsafe.Pointer(devicesC))

	if !C.WrapGetSupportedEvaluatorTypes(m.handler, &devicesC, (*C.size_t)(&devicesNum)) {
		return nil, fmt.Errorf(formatErrorMessage, ErrGetDevices, GetError())
	}

	devicesCTmp := (*[1 << 28]C.int)(unsafe.Pointer(devicesC))[:devicesNum:devicesNum]

	devices := make([]EvaluatorType, 0, len(devicesCTmp))
	for _, d := range devicesCTmp {
		devices = append(devices, EvaluatorType(d))
	}
	return devices, nil
}

// EnableGPUEvaluation set use CUDA GPU device for model evaluation.
// Only device 0 is supported for "now"
// See more details https://github.com/catboost/catboost/issues/2774
func (m *Model) EnableGPUEvaluation() error {
	if runtime.GOOS != "linux" {
		return ErrNotSupportedGPU
	}

	devices, err := m.GetSupportedEvaluatorTypes()
	if err != nil {
		return err
	}

	if !slices.Contains(devices, GPU) {
		return ErrNotSupportedGPU
	}

	deviceID := 0

	if !C.WrapEnableGPUEvaluation(m.handler, C.int(deviceID)) {
		return ErrEnabledGPU
	}

	return nil
}

// GetModelUsedFeaturesNames returns names of features used in the model.
func (m *Model) GetModelUsedFeaturesNames() ([]string, error) {
	featuresCount := m.GetFeaturesCount()

	featuresC := C.makeCharArray1D(C.int(featuresCount))
	defer C.freeCharArray1D(featuresC, C.int(featuresCount))

	featuresCountC := C.size_t(featuresCount)
	if !C.WrapGetModelUsedFeaturesNames(m.handler, &featuresC, &featuresCountC) {
		return nil, fmt.Errorf(formatErrorMessage, ErrGetModelUsedFeaturesNames, GetError())
	}

	features := make([]string, 0, featuresCount)

	// https://go.dev/wiki/cgo#turning-c-arrays-into-go-slices
	featuresTmpC := (*[1 << 28]*C.char)(unsafe.Pointer(featuresC))[:featuresCount:featuresCount]

	for _, featureC := range featuresTmpC {
		features = append(features, C.GoString(featureC))
	}

	return features, nil
}

// GetFloatFeaturesCount returns expected float feature count for model.
func (m *Model) GetFloatFeaturesCount() int {
	return int(C.WrapGetFloatFeaturesCount(m.handler))
}

// GetCatFeaturesCount returns expected categorical feature count for model.
func (m *Model) GetCatFeaturesCount() int {
	return int(C.WrapGetCatFeaturesCount(m.handler))
}

// GetFeaturesCount returns all expected feature count for model.
func (m *Model) GetFeaturesCount() int {
	return m.GetCatFeaturesCount() + m.GetFloatFeaturesCount()
}

// GetDimensionsCount returns number of dimensions in model.
func (m *Model) GetDimensionsCount() int {
	return int(C.WrapGetDimensionsCount(m.handler))
}

// GetRowResultSize return size row result.
func (m *Model) GetRowResultSize() int {
	if m.predictionType == Class {
		return 1
	}

	return m.GetDimensionsCount()
}

// Predict returns predictions.
func (m *Model) Predict(floats [][]float32, cats [][]string) ([]float64, error) {
	var nSamples int

	// Get length sample
	nSamples = len(floats)
	if nSamples == 0 {
		nSamples = len(cats)
	}

	floatFeaturesCount := m.GetFloatFeaturesCount()
	catFeaturesCount := m.GetCatFeaturesCount()

	// Special for Multiclassification (size > 1)
	size := m.GetRowResultSize()

	preds := make([]float64, nSamples*size)

	floatsC := makeFloatArray2D(floats)
	defer C.free(unsafe.Pointer(floatsC))

	catsC := makeCharArray2D(cats)
	defer C.freeCharArray2D(catsC, C.int(nSamples), C.int(catFeaturesCount))

	if !C.WrapCalcModelPrediction(
		m.handler,
		C.size_t(nSamples),
		floatsC,
		C.size_t(floatFeaturesCount),
		catsC,
		C.size_t(catFeaturesCount),
		(*C.double)(&preds[0]),
		C.size_t(len(preds)),
	) {
		return nil, fmt.Errorf(formatErrorMessage, ErrCalcModelPrediction, GetError())
	}

	return preds, nil
}

// PredictSingle returns prediction.
func (m *Model) PredictSingle(floats []float32, cats []string) ([]float64, error) {
	catsC := makeCharArray1D(cats)
	defer C.freeCharArray1D(catsC, C.int(len(cats)))

	size := m.GetRowResultSize()
	preds := make([]float64, 1*size)

	floatsC := new(C.float)
	if len(floats) > 0 {
		floatsC = (*C.float)(&floats[0])
	}

	if !C.WrapCalcModelPredictionSingle(
		m.handler,
		floatsC,
		C.size_t(len(floats)),
		catsC,
		C.size_t(len(cats)),
		(*C.double)(&preds[0]),
		C.size_t(len(preds))) {
		return nil, GetError()
	}

	return preds, nil
}

// Transform change data for result Multiclassification.
func (m *Model) Transform(preds []float64) [][]float64 {
	size := m.GetRowResultSize()
	result := make([][]float64, 0, len(preds)/size)

	for i := 0; i < len(preds); i += size {
		pred := make([]float64, 0, size)
		for j := i; j < i+size; j++ {
			pred = append(pred, preds[j])
		}
		result = append(result, pred)
	}

	return result
}

// GetCatFeatureIndices expected indices of category features used in the model.
func (m *Model) GetCatFeatureIndices() ([]uint64, error) {
	catsFeatureNum := uint64(m.GetCatFeaturesCount())
	if catsFeatureNum == 0 {
		return []uint64{}, nil
	}

	catsFeatureIndices := make([]*uint64, catsFeatureNum)
	catsFeatureIndicesC := (*C.size_t)(catsFeatureIndices[0])
	defer C.free(unsafe.Pointer(catsFeatureIndicesC))

	if !C.WrapGetCatFeatureIndices(m.handler, &catsFeatureIndicesC, (*C.size_t)(&catsFeatureNum)) {
		return nil, fmt.Errorf(formatErrorMessage, ErrGetIndices, GetError())
	}

	indices := (*[1 << 28]uint64)(unsafe.Pointer(catsFeatureIndicesC))[:catsFeatureNum:catsFeatureNum]
	return indices, nil
}

// GetFloatFeatureIndices expected indices of float features used in the model.
func (m *Model) GetFloatFeatureIndices() ([]uint64, error) {
	floatsFeatureNum := uint64(m.GetFloatFeaturesCount())
	if floatsFeatureNum == 0 {
		return []uint64{}, nil
	}

	floatsFeatureIndices := make([]*uint64, floatsFeatureNum)
	floatsFeatureIndicesC := (*C.size_t)(floatsFeatureIndices[0])
	defer C.free(unsafe.Pointer(floatsFeatureIndicesC))

	if !C.WrapGetFloatFeatureIndices(m.handler, &floatsFeatureIndicesC, (*C.size_t)(&floatsFeatureNum)) {
		return nil, fmt.Errorf(formatErrorMessage, ErrGetIndices, GetError())
	}

	indices := (*[1 << 28]uint64)(unsafe.Pointer(floatsFeatureIndicesC))[:floatsFeatureNum:floatsFeatureNum]
	return indices, nil
}

// getFromLibraryFn retruns point to function from CatBoost shared memory.
func getFromLibraryFn(handle unsafe.Pointer, fnName string) unsafe.Pointer {
	cFnName := C.CString(fnName)
	defer C.free(unsafe.Pointer(cFnName))

	fn := C.dlsym(handle, cFnName)
	if fn == nil {
		C.dlclose(handle)
		msg := C.GoString(C.dlerror())
		panic(fmt.Sprintf("Error looking up %s in `%s`: %s", fnName, catboostSharedLibraryPath, msg))
	}

	return fn
}

// GetError returns last error from model.
// If error ocured will return stored exception message.
// If no error ocured, will return invalid pointer.
func GetError() error {
	messageC := C.WrapGetErrorString()
	message := C.GoString(messageC)

	i := strings.Index(message, "catboost.git")
	if i == -1 {
		return nil
	}

	return errors.New(message[i:])
}

// Helper for create convert [][]string to `C`.
func makeCharArray2D(cats [][]string) ***C.char {
	nSamples := len(cats)
	catsC := C.makeCharArray2D(C.int(nSamples))

	for i, cat := range cats {
		catC := C.makeCharArray1D(C.int(len(cat)))
		for i, c := range cat {
			C.setCharArray1D(catC, C.CString(c), C.int(i))
		}
		C.setCharArray2D(catsC, catC, C.int(i))
	}

	return catsC
}

// Helper for create convert []string to `C`.
func makeCharArray1D(cats []string) **C.char {
	nSamples := len(cats)
	catsC := C.makeCharArray1D(C.int(nSamples))

	for i := range cats {
		C.setCharArray1D(catsC, C.CString(cats[i]), C.int(i))
	}

	return catsC
}

// Helper for create convert [][]float32 to `C`.
func makeFloatArray2D(floats [][]float32) **C.float {
	nSamples := len(floats)
	floatsC := C.makeFloatArray2D(C.int(nSamples))

	for i, v := range floats {
		C.setFloatArray2D(floatsC, (*C.float)(&v[0]), C.int(i))
	}

	return floatsC
}
