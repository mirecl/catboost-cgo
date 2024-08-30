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
	modelPath := path.Join(filepath.Dir(fileName), "uncertainty.cbm")

	// Initialize CatBoostRegressor
	model, err := cb.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Get batch predicted RMSEWithUncertainty
	model.SetPredictionType(cb.RMSEWithUncertainty)

	// Initialize data
	floats := [][]float32{}
	cats := [][]string{{"0", "0"}}

	// Get batch predicted RMSEWithUncertainty
	preds, err := model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds `RMSEWithUncertainty`: %v\n", preds)
}
