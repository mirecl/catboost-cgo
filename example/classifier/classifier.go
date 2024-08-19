package main

import (
	"fmt"
	"log"
	"path"
	"path/filepath"
	"runtime"

	cb "github.com/mirecl/catboost-cgo/catboost"
)

func main() {
	_, fileName, _, _ := runtime.Caller(0)
	modelPath := path.Join(filepath.Dir(fileName), "classifier.cbm")

	// Initialize CatBoostClassifier
	model, err := cb.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Initialize data
	floats := [][]float32{{2, 4, 6, 8, 5}, {1, 4, 50, 60, 5}}
	cats := [][]string{{"a", "b"}, {"a", "d"}}

	// Get batch predicted RawFormulaVal
	preds, err := model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds `RawFormulaVal`: %.8f\n", preds)

	// Get single predicted RawFormulaVal
	pred, err := model.PredictSingle(floats[0], cats[0])
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Pred `RawFormulaVal`: %.8f\n", pred)

	// Get batch predicted Probability
	model.SetPredictionType(cb.Probablity)
	preds, err = model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds `Probability`: %v\n", preds)

	// Get single predicted Probability
	pred, err = model.PredictSingle(floats[0], cats[0])
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Pred `Probability`: %.8f\n", pred)

}
