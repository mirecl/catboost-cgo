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
	modelPath := path.Join(filepath.Dir(fileName), "titanic.cbm")

	// Initialize CatBoostClassifier
	model, err := cb.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Initialize data
	floats := [][]float32{
		{34.5, 7.8292},
		{47.0, 7.0},
		{62.0, 9.6875},
		{27.0, 8.6625},
		{22.0, 12.2875},
	}
	cats := [][]string{
		{"892", "3", "Kelly, Mr. James", "male", "0", "0", "330911", "-999", "Q"},
		{"893", "3", "Wilkes, Mrs. James (Ellen Needs)", "female", "1", "0", "363272", "-999", "S"},
		{"894", "2", "Myles, Mr. Thomas Francis", "male", "0", "0", "240276", "-999", "Q"},
		{"895", "3", "Wirz, Mr. Albert", "male", "0", "0", "315154", "-999", "S"},
		{"896", "3", "Hirvonen, Mrs. Alexander (Helga E Lindqvist)", "female", "1", "1", "3101298", "-999", "S"},
	}

	// Get batch predicted Class
	model.SetPredictionType(cb.Class)
	preds, err := model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds `Class`: %.0f\n", preds)

	// Get batch predicted Probability
	model.SetPredictionType(cb.Probablity)
	preds, err = model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds `Probability`: %v\n", preds)
}
