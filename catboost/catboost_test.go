package catboost_test

import (
	"errors"
	"fmt"
	"runtime"
	"testing"

	cb "github.com/mirecl/catboost-cgo/catboost"
	"github.com/stretchr/testify/require"
)

const (
	testModelPathRegressor           = "../example/regressor/regressor.cbm"
	testModelPathClassifier          = "../example/classifier/classifier.cbm"
	testModelPathMulticlassification = "../example/multiclassification/multiclassification.cbm"
	testModelPathMetadata            = "../example/metadata/metadata.cbm"
)

func TestVersion(t *testing.T) {
	require.Equal(t, "v1.2.7", cb.Version())
}

func TestFeatureIndices(t *testing.T) {
	modelClassifier, err := cb.LoadFullModelFromFile(testModelPathClassifier)
	require.NoError(t, err)
	require.NotNil(t, modelClassifier)

	catIndices, err := modelClassifier.GetCatFeatureIndices()
	require.NoError(t, err)
	require.Equal(t, []uint64{0, 1}, catIndices)

	floatIndices, err := modelClassifier.GetFloatFeatureIndices()
	require.NoError(t, err)
	require.Equal(t, []uint64{2, 3, 4, 5}, floatIndices)
}

func TestLoadFullModel(t *testing.T) {
	modelRegressor, err := cb.LoadFullModelFromFile(testModelPathRegressor)
	require.NoError(t, err)
	require.NotNil(t, modelRegressor)

	modelFake, err := cb.LoadFullModelFromFile("fake.cbm")
	require.ErrorIs(t, err, cb.ErrLoadFullModelFromFile)
	require.Nil(t, modelFake)
}

func TestPredict(t *testing.T) {
	modelRegressor, err := cb.LoadFullModelFromFile(testModelPathRegressor)
	require.NoError(t, err)
	require.NotNil(t, modelRegressor)

	modelClassifier, err := cb.LoadFullModelFromFile(testModelPathClassifier)
	require.NoError(t, err)
	require.NotNil(t, modelClassifier)

	modelMulticlassification, err := cb.LoadFullModelFromFile(testModelPathMulticlassification)
	require.NoError(t, err)
	require.NotNil(t, modelClassifier)

	testCases := []struct {
		model       *cb.Model
		predictType cb.PredictionType
		floats      [][]float32
		cats        [][]string
		preds       []float64
		pred        []float64
	}{
		{
			model:       modelRegressor,
			predictType: cb.RawFormulaVal,
			floats:      [][]float32{{2, 4, 6, 8}, {1, 4, 50, 60}},
			cats:        [][]string{{}, {}},
			preds:       []float64{15.625, 18.125},
			pred:        []float64{15.625},
		},
		{
			model:       modelClassifier,
			predictType: cb.Class,
			floats:      [][]float32{{2, 4, 6, 8, 5}, {1, 4, 50, 60, 5}},
			cats:        [][]string{{"a", "b"}, {"a", "d"}},
			preds:       []float64{1, 1},
			pred:        []float64{1},
		},
		{
			model:       modelClassifier,
			predictType: cb.Probablity,
			floats:      [][]float32{{2, 4, 6, 8, 5}, {1, 4, 50, 60, 5}},
			cats:        [][]string{{"a", "b"}, {"a", "d"}},
			preds:       []float64{0.629855013297618, 0.5358421019868945},
			pred:        []float64{0.629855013297618},
		},
		{
			model:       modelMulticlassification,
			predictType: cb.Class,
			floats:      [][]float32{{1996, 197}, {1968, 37}, {2002, 77}, {1948, 59}},
			cats:        [][]string{{"winter"}, {"winter"}, {"summer"}, {"summer"}},
			preds:       []float64{2, 2, 1, 2},
			pred:        []float64{2},
		},
		{
			model:       modelMulticlassification,
			predictType: cb.Probablity,
			floats:      [][]float32{{1996, 197}, {1968, 37}},
			cats:        [][]string{{"winter"}, {"winter"}},
			preds:       []float64{0.2006095939361826, 0.2862616005077138, 0.5131288055561035, 0.07388963079437862, 0.060717262866699366, 0.8653931063389221},
			pred:        []float64{0.2006095939361826, 0.2862616005077138, 0.5131288055561035},
		},
	}

	for i, testCase := range testCases {
		label := fmt.Sprintf("testCase[%d]", i)
		t.Run(label, func(t *testing.T) {
			err := testCase.model.SetPredictionType(testCase.predictType)
			require.NoError(t, err)

			preds, err := testCase.model.Predict(testCase.floats, testCase.cats)
			require.NoError(t, err)
			require.Equal(t, testCase.preds, preds)

			preds, err = testCase.model.PredictSingle(testCase.floats[0], testCase.cats[0])
			require.NoError(t, err)
			require.Equal(t, testCase.pred, preds)
		})
	}
}

func TestTransform(t *testing.T) {
	modelMulticlassification, err := cb.LoadFullModelFromFile(testModelPathMulticlassification)
	require.NoError(t, err)
	require.NotNil(t, modelMulticlassification)

	preds := []float64{1, 2, 3, 4, 5, 6}

	err = modelMulticlassification.SetPredictionType(cb.Probablity)
	require.NoError(t, err)

	result := modelMulticlassification.Transform(preds)
	require.Equal(t, [][]float64{{1, 2, 3}, {4, 5, 6}}, result)
}

func TestMetadata(t *testing.T) {
	modelMetadata, err := cb.LoadFullModelFromFile(testModelPathMetadata)
	require.NoError(t, err)
	require.NotNil(t, modelMetadata)

	featuresNames, err := modelMetadata.GetModelUsedFeaturesNames()
	require.NoError(t, err)

	require.Equal(t, []string{"Column=0", "Column=1", "Column=2", "Column=3", "Column=4", "Column=5", "Column=6", "Column=7", "Column=8", "Column=9", "CatColumn_1", "CatColumn_2"}, featuresNames)
	require.Equal(t, 2, modelMetadata.GetCatFeaturesCount())
	require.Equal(t, 10, modelMetadata.GetFloatFeaturesCount())
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(cb.MetaModelGUID))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(cb.MetaOutputOptions))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(cb.MetaParams))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(cb.MetaTrainFinishTime))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(cb.MetaTraining))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(cb.MetaVersionInfo))
}

func TestGetError(t *testing.T) {
	// init test model
	_, err := cb.LoadFullModelFromFile(testModelPathMetadata)
	require.NoError(t, err)

	err = cb.GetError()
	require.NoError(t, err)
}

func TestGetSupportedEvaluatorTypes(t *testing.T) {
	// init test model
	model, err := cb.LoadFullModelFromFile(testModelPathRegressor)
	require.NoError(t, err)

	devices, err := model.GetSupportedEvaluatorTypes()
	require.NoError(t, err)
	require.True(t, len(devices) > 0)
}

func TestEnableGPUEvaluation(t *testing.T) {
	// init test model
	model, err := cb.LoadFullModelFromFile(testModelPathRegressor)
	require.NoError(t, err)

	err = model.EnableGPUEvaluation()

	if runtime.GOOS == "darwin" {
		require.True(t, errors.Is(err, cb.ErrNotSupportedGPU))
	}

	if runtime.GOOS == "linux" {
		require.True(t, errors.Is(err, cb.ErrEnabledGPU))
	}
}
