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

type PredictionType string

const (
	RawFormulaVal       PredictionType = "RawFormulaVal"
	Probablity          PredictionType = "Probability"
	Class               PredictionType = "Class"
	RMSEWithUncertainty PredictionType = "RMSEWithUncertainty"
	Exponent            PredictionType = "Exponent"
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
	ErrGetModelUsedFeaturesNames = errors.New("failed get used features name")
	ErrCalcModelPrediction       = errors.New("failed inference model")
	ErrNotSupported              = errors.New("supported only Linux and MacOS")
	ErrLoadLibrary               = errors.New("failed loading CatBoost shared library")
	ErrSetPredictionType         = errors.New("failed set prediction type")
)

var catboostSharedLibraryPath = ""

func Version() string {
	return fmt.Sprintf("v%d.%d.%d", C.CATBOOST_APPLIER_MAJOR, C.CATBOOST_APPLIER_MINOR, C.CATBOOST_APPLIER_FIX)
}

func SetSharedLibraryPath(path string) {
	catboostSharedLibraryPath = path
}

func initialization() error {
	if !checkPlatform() {
		return ErrNotSupported
	}

	initSharedLibraryPath()

	cName := C.CString(catboostSharedLibraryPath)
	defer C.free(unsafe.Pointer(cName))

	handle := C.dlopen(cName, C.RTLD_LAZY)
	if handle == nil {
		C.dlclose(handle)
		msg := C.GoString(C.dlerror())
		return fmt.Errorf("%w `%s`: %s", ErrLoadLibrary, catboostSharedLibraryPath, msg)
	}

	l := library{handle}

	// Load function from CatBoost shared library
	l.RegisterFn("ModelCalcerCreate")
	l.RegisterFn("LoadFullModelFromBuffer")
	l.RegisterFn("CalcModelPredictionSingle")
	l.RegisterFn("CalcModelPrediction")
	l.RegisterFn("GetErrorString")
	l.RegisterFn("GetFloatFeaturesCount")
	l.RegisterFn("GetCatFeaturesCount")
	l.RegisterFn("GetDimensionsCount")
	l.RegisterFn("SetPredictionTypeString")
	l.RegisterFn("GetModelUsedFeaturesNames")
	l.RegisterFn("GetModelInfoValue")

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
	default:
		panic(fmt.Sprintf("not supported function from catboost library: %s", fnName))
	}
}

func initSharedLibraryPath() {
	if catboostSharedLibraryPath != "" {
		return
	}

	if catboostSharedLibraryPath = os.Getenv("CATBOOST_LIBRARY_PATH"); catboostSharedLibraryPath != "" {
		return
	}

	catboostSharedLibraryPath = fmt.Sprintf("/usr/local/lib/libcatboostmodel.%s", getExt())
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
		return fmt.Errorf("%w `%s`: %s", ErrSetPredictionType, p, GetError())
	}

	m.predictionType = p

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

	if !C.WrapCalcModelPredictionSingle(
		m.handler,
		(*C.float)(&floats[0]),
		C.size_t(len(floats)),
		catsC,
		C.size_t(len(cats)),
		(*C.double)(&preds[0]),
		C.size_t(len(preds))) {
		return nil, fmt.Errorf("%s", GetError())
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
// If no error ocured, will return invalid pointer
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
