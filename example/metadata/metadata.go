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
	modelPath := path.Join(filepath.Dir(fileName), "metadata.cbm")

	model, err := catboost.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	catboostVersinInfo := model.GetModelInfoValue(catboost.MetaVersionInfo)
	fmt.Printf("CATBOOST_VERSION_INFO:\n%s\n", catboostVersinInfo)

	trainFinishTime := model.GetModelInfoValue(catboost.MetaTrainFinishTime)
	fmt.Printf("TRAIN_FINISH_TIME:\n%s\n\n", trainFinishTime)

	outputOptions := model.GetModelInfoValue(catboost.MetaOutputOptions)
	fmt.Printf("OUTPUT_OPTIONS:\n%s\n\n", outputOptions)

	exampleValue := model.GetModelInfoValue("example_key")
	fmt.Printf("EXAMPLE_KEY:\n%s\n\n", exampleValue)

	modelGuid := model.GetModelInfoValue(catboost.MetaModelGuid)
	fmt.Printf("MODEL_GUID:\n%s\n\n", modelGuid)

	params := model.GetModelInfoValue(catboost.MetaParams)
	fmt.Printf("PARAMS:\n%s\n\n", params)

	fmt.Printf("Float features count: %d\n", model.GetFloatFeaturesCount())
	fmt.Printf("Cat features count: %d\n", model.GetCatFeaturesCount())

	features, err := model.GetModelUsedFeaturesNames()
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Used Features names: %v\n", features)
}
