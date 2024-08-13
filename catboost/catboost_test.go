package catboost_test

import (
	"fmt"
	"testing"

	"github.com/mirecl/catboost-cgo/catboost"
	"github.com/stretchr/testify/require"
)

const (
	testModelPathRegressor           = "../example/regressor/regressor.cbm"
	testModelPathClassifier          = "../example/classifier/classifier.cbm"
	testModelPathMulticlassification = "../example/multiclassification/multiclassification.cbm"
	testModelPathMetadata            = "../example/metadata/metadata.cbm"
)

func TestLoadFullModel(t *testing.T) {
	modelRegressor, err := catboost.LoadFullModelFromFile(testModelPathRegressor)
	require.NoError(t, err)
	require.NotNil(t, modelRegressor)

	modelFake, err := catboost.LoadFullModelFromFile("fake.cbm")
	require.ErrorIs(t, err, catboost.ErrLoadFullModelFromFile)
	require.Nil(t, modelFake)

	catboost.SetSharedLibraryPath("fake.so")
	require.PanicsWithValue(t, "Error loading CatBoost shared library `fake.so`: dlclose(0x0): invalid handle", func() {
		_, _ = catboost.LoadFullModelFromFile(testModelPathRegressor)
	})
	catboost.SetSharedLibraryPath("")

	b := []byte("0")
	model, err := catboost.LoadFullModelFromBuffer(b)
	require.ErrorIs(t, err, catboost.ErrLoadFullModelFromBuffer)
	require.Nil(t, model)
}

func TestPredict(t *testing.T) {
	modelRegressor, err := catboost.LoadFullModelFromFile(testModelPathRegressor)
	require.NoError(t, err)
	require.NotNil(t, modelRegressor)

	modelClassifier, err := catboost.LoadFullModelFromFile(testModelPathClassifier)
	require.NoError(t, err)
	require.NotNil(t, modelClassifier)

	modelMulticlassification, err := catboost.LoadFullModelFromFile(testModelPathMulticlassification)
	require.NoError(t, err)
	require.NotNil(t, modelClassifier)

	testCases := []struct {
		model       *catboost.Model
		predictType catboost.PredictionType
		floats      [][]float32
		cats        [][]string
		preds       []float64
		pred        []float64
	}{
		{
			model:       modelRegressor,
			predictType: catboost.RawFormulaVal,
			floats:      [][]float32{{2, 4, 6, 8}, {1, 4, 50, 60}},
			cats:        [][]string{{}, {}},
			preds:       []float64{15.625, 18.125},
			pred:        []float64{15.625},
		},
		{
			model:       modelClassifier,
			predictType: catboost.Class,
			floats:      [][]float32{{2, 4, 6, 8, 5}, {1, 4, 50, 60, 5}},
			cats:        [][]string{{"a", "b"}, {"a", "d"}},
			preds:       []float64{1, 1},
			pred:        []float64{1},
		},
		{
			model:       modelClassifier,
			predictType: catboost.Probablity,
			floats:      [][]float32{{2, 4, 6, 8, 5}, {1, 4, 50, 60, 5}},
			cats:        [][]string{{"a", "b"}, {"a", "d"}},
			preds:       []float64{0.629855013297618, 0.5358421019868945},
			pred:        []float64{0.629855013297618},
		},
		{
			model:       modelMulticlassification,
			predictType: catboost.Class,
			floats:      [][]float32{{1996, 197}, {1968, 37}, {2002, 77}, {1948, 59}},
			cats:        [][]string{{"winter"}, {"winter"}, {"summer"}, {"summer"}},
			preds:       []float64{2, 2, 1, 2},
			pred:        []float64{2},
		},
		{
			model:       modelMulticlassification,
			predictType: catboost.Probablity,
			floats:      [][]float32{{1996, 197}, {1968, 37}},
			cats:        [][]string{{"winter"}, {"winter"}},
			preds:       []float64{0.2006095939361826, 0.2862616005077138, 0.5131288055561035, 0.07388963079437862, 0.060717262866699366, 0.8653931063389221},
			pred:        []float64{0.2006095939361826, 0.2862616005077138, 0.5131288055561035},
		},
	}

	for i, testCase := range testCases {
		label := fmt.Sprintf("testCase[%d]", i)
		t.Run(label, func(t *testing.T) {
			testCase.model.SetPredictionType(testCase.predictType)

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
	modelMulticlassification, err := catboost.LoadFullModelFromFile(testModelPathMulticlassification)
	require.NoError(t, err)
	require.NotNil(t, modelMulticlassification)

	preds := []float64{1, 2, 3, 4, 5, 6}

	modelMulticlassification.SetPredictionType(catboost.Probablity)
	result := modelMulticlassification.Transform(preds)
	require.Equal(t, [][]float64{{1, 2, 3}, {4, 5, 6}}, result)
}

func TestMetadata(t *testing.T) {
	modelMetadata, err := catboost.LoadFullModelFromFile(testModelPathMetadata)
	require.NoError(t, err)
	require.NotNil(t, modelMetadata)

	featuresNames, err := modelMetadata.GetModelUsedFeaturesNames()
	require.NoError(t, err)

	require.Equal(t, []string{"Column=0", "Column=1", "Column=2", "Column=3", "Column=4", "Column=5", "Column=6", "Column=7", "Column=8", "Column=9", "CatColumn_1", "CatColumn_2"}, featuresNames)
	require.Equal(t, 2, modelMetadata.GetCatFeaturesCount())
	require.Equal(t, 10, modelMetadata.GetFloatFeaturesCount())
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(catboost.MetaModelGuid))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(catboost.MetaOutputOptions))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(catboost.MetaParams))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(catboost.MetaTrainFinishTime))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(catboost.MetaTraining))
	require.NotEmpty(t, modelMetadata.GetModelInfoValue(catboost.MetaVersionInfo))
}
