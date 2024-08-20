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
	modelPath := path.Join(filepath.Dir(fileName), "regressor.cbm")

	// Initialize CatBoostRegressor
	model, err := cb.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Initialize data
	floats := [][]float32{{2, 4, 6, 8}, {1, 4, 50, 60}}
	cats := [][]string{{}}

	// Get batch predictions
	preds, err := model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds `RawFormulaVal`: %.3f\n", preds)

	// Get single predictions
	pred, err := model.PredictSingle(floats[0], cats[0])
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Pred `RawFormulaVal`: %.3f\n", pred)
}
