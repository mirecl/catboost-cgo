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
	modelPath := path.Join(filepath.Dir(fileName), "survival.cbm")

	// Initialize CatBoostRegressor
	model, err := cb.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Get batch predicted Exponent
	model.SetPredictionType(cb.Exponent)

	// Initialize data
	floats := [][]float32{
		{60.0, 20.4684, 52.0, 84.0, 10.0, 169.0},
		{61.0, 25.4607, 80.0, 111.0, 5.0, 130.0},
		{85.0, 21.94843, 104.0, 97.0, 9.0, 198.0},
	}

	cats := [][]string{
		{"0", "0", "0", "1", "1", "0", "0", "0"},
		{"0", "0", "1", "0", "1", "0", "0", "0"},
		{"0", "0", "0", "1", "1", "1", "0", "0"}}

	// Get batch predicted Exponent
	preds, err := model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds `Exponent`: %v\n", preds)
}
