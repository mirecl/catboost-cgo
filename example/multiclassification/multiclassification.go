package main

import (
	"fmt"
	"log"
	"path"
	"path/filepath"
	"runtime"

	"github.com/mirecl/catboost-cgo/catboost"
)

func main() {
	_, fileName, _, _ := runtime.Caller(0)
	modelPath := path.Join(filepath.Dir(fileName), "multiclassification.cbm")

	// Initialize CatBoostClassifier
	model, err := catboost.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Initialize data
	floats := [][]float32{{1996, 197}, {1968, 37}, {2002, 77}, {1948, 59}}
	cats := [][]string{{"winter"}, {"winter"}, {"summer"}, {"summer"}}

	// Get batch predicted RawFormulaVal
	preds, err := model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}

	predsMulti := model.Transform(preds)
	fmt.Printf("Preds `RawFormulaVal`: %.8f\n", predsMulti)

	// Get single predicted RawFormulaVal
	pred, err := model.PredictSingle(floats[0], cats[0])
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Pred `RawFormulaVal`: %.8f\n", pred)

	// Get batch predicted probabilities for each class
	model.SetPredictionType(catboost.Probablity)
	preds, err = model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}

	predsMulti = model.Transform(preds)
	fmt.Printf("Preds `Probability`: %.8f\n", predsMulti)

	// Get single predicted probabilities for each class
	pred, err = model.PredictSingle(floats[0], cats[0])
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Pred `Probability`: %.8f\n", pred)

	// Get batch predicted classes
	model.SetPredictionType(catboost.Class)
	preds, err = model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}

	predsMulti = model.Transform(preds)
	fmt.Printf("Preds `Class`: %.0f\n", predsMulti)

	// Get single predicted classes
	pred, err = model.PredictSingle(floats[0], cats[0])
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Pred `Class`: %.0f\n", pred)
}
