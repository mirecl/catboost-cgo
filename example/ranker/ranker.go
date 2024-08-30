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
	modelPath := path.Join(filepath.Dir(fileName), "ranker.cbm")

	// Initialize CatBoostRanker
	model, err := cb.LoadFullModelFromFile(modelPath)
	if err != nil {
		log.Fatalln(err)
	}

	// Initialize data
	floats := [][]float32{{
		2.0000000e00, 0.0000000e00, 2.0000000e00, 1.0000000e00, 2.0000000e00, 1.0000000e00, 0.0000000e00,
		1.0000000e00, 5.0000000e-01, 1.0000000e00, 3.1000000e01, 0.0000000e00, 1.1000000e01, 7.0000000e00,
		4.9000000e01, 6.5531250e00, 1.5011174e01, 1.2950828e01, 1.4369216e01, 6.5508690e00, 4.0000000e00,
		0.0000000e00, 2.0000000e00, 1.0000000e00, 7.0000000e00, 2.0000000e00, 0.0000000e00, 1.0000000e00,
		0.0000000e00, 3.0000000e00, 2.0000000e00, 0.0000000e00, 1.0000000e00, 1.0000000e00, 4.0000000e00,
		2.0000000e00, 0.0000000e00, 1.0000000e00, 5.0000000e-01, 3.5000000e00, 0.0000000e00, 0.0000000e00,
		0.0000000e00, 2.5000000e-01, 2.5000000e-01, 1.2903200e-01, 0.0000000e00, 1.8181800e-01, 1.4285700e-01,
		1.4285700e-01, 6.4516000e-02, 0.0000000e00, 9.0909000e-02, 0.0000000e00, 6.1224000e-02, 6.4516000e-02,
		0.0000000e00, 9.0909000e-02, 1.4285700e-01, 8.1633000e-02, 6.4516000e-02, 0.0000000e00, 9.0909000e-02,
		7.1429000e-02, 7.1429000e-02, 0.0000000e00, 0.0000000e00, 0.0000000e00, 5.1020000e-03, 1.0400000e-04,
		1.3106251e01, 0.0000000e00, 1.2950828e01, 6.8290930e00, 2.2821554e01, 6.3401830e00, 0.0000000e00,
		6.1298360e00, 0.0000000e00, 1.0145764e01, 6.7660680e00, 0.0000000e00, 6.8209920e00, 6.8290930e00,
		1.2675790e01, 6.5531250e00, 0.0000000e00, 6.4754140e00, 3.4145460e00, 1.1410777e01, 4.5344000e-02,
		0.0000000e00, 1.1942400e-01, 1.1659127e01, 1.6002580e00, 1.0000000e00, 0.0000000e00, 1.0000000e00,
		0.0000000e00, 1.0000000e00, 1.0000000e00, 0.0000000e00, 1.0000000e00, 6.7132900e-01, 9.8981100e-01,
		1.7818264e01, 0.0000000e00, 1.0183562e01, 7.6338160e00, 1.9436549e01, -6.3404310e00, -1.2071142e01,
		-7.1911410e00, -1.3131176e01, -5.7551620e00, -1.3631532e01, -1.6095443e01, -1.4367199e01, -1.6975368e01,
		-1.2639740e01, -5.6920090e00, -1.2919850e01, -5.0058500e00, -1.3980776e01, -5.5091020e00, 2.0000000e00,
		3.5000000e01, 1.0000000e00, 0.0000000e00, 2.6600000e02, 2.5070000e04, 2.8000000e01, 7.0000000e00,
		0.0000000e00, 0.0000000e00, 0.0000000e00},
	}

	cats := [][]string{}

	// Get batch predicted Exponent
	preds, err := model.Predict(floats, cats)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("Preds: %v\n", preds)
}
