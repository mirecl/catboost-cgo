package main

import (
	"fmt"
	"log"

	cb "github.com/mirecl/catboost-cgo/catboost"
)

func main() {
	// Load model trained with text features (see text.py)
	model, err := cb.LoadFullModelFromFile("example/text/text.cbm")
	if err != nil {
		log.Fatalf("LoadFullModelFromFile: %v", err)
	}
	defer model.Delete()

	// Inspect text feature metadata
	fmt.Printf("Text feature count : %d\n", model.GetTextFeaturesCount())
	fmt.Printf("Float feature count: %d\n", model.GetFloatFeaturesCount())
	fmt.Printf("Total feature count: %d\n", model.GetFeaturesCount())

	textIndices, err := model.GetTextFeatureIndices()
	if err != nil {
		log.Fatalf("GetTextFeatureIndices: %v", err)
	}
	fmt.Printf("Text feature indices: %v\n", textIndices)

	// Eval samples matching text.py:
	//   ["amazing value", 4.6, 100.0]
	//   ["poor quality",  1.5,  35.0]
	floats := [][]float32{
		{4.6, 100.0},
		{1.5, 35.0},
	}
	cats := [][]string{{}, {}}
	texts := [][]string{
		{"amazing value"},
		{"poor quality"},
	}

	// Batch prediction
	batchPreds, err := model.PredictText(floats, cats, texts)
	if err != nil {
		log.Fatalf("PredictText: %v", err)
	}
	fmt.Println("Batch RawFormulaVal predictions:")
	for i, p := range batchPreds {
		fmt.Printf("  sample[%d] = %.10f\n", i, p)
	}

	// Single-sample prediction (first eval sample)
	singlePreds, err := model.PredictSingleText(
		floats[0],
		cats[0],
		texts[0],
	)
	if err != nil {
		log.Fatalf("PredictSingleText: %v", err)
	}
	fmt.Println("Single RawFormulaVal prediction (sample[0]):")
	for i, p := range singlePreds {
		fmt.Printf("  result[%d] = %.10f\n", i, p)
	}
}
