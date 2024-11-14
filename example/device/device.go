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

	devices, err := model.GetSupportedEvaluatorTypes()
	if err != nil {
		log.Fatalln(err)
	}

	for _, device := range devices {
		switch device {
		case cb.CPU:
			fmt.Println("Supported CPU")
		case cb.GPU:
			fmt.Println("Supported GPU")
		default:
			fmt.Println("Unknown device")
		}
	}
}
