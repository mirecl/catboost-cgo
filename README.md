[![PkgGoDev](https://pkg.go.dev/badge/mod/github.com/mirecl/catboost-cgo)](https://pkg.go.dev/mod/github.com/mirecl/catboost-cgo) ![GitHub Release](https://img.shields.io/github/v/release/mirecl/catboost-cgo)
![](https://img.shields.io/github/go-mod/go-version/mirecl/catboost-cgo) [![CI](https://github.com/mirecl/catboost-cgo/actions/workflows/ci.yml/badge.svg)](https://github.com/mirecl/catboost-cgo/actions/workflows/ci.yml) [![codecov](https://codecov.io/github/mirecl/catboost-cgo/graph/badge.svg?token=HUG4WWSSZN)](https://codecov.io/github/mirecl/catboost-cgo)
[![Go Report Card](https://goreportcard.com/badge/github.com/mirecl/catboost-cgo)](https://goreportcard.com/report/github.com/mirecl/catboost-cgo) [![Mentioned in Awesome Go](https://awesome.re/mentioned-badge.svg)](https://github.com/avelino/awesome-go)  \
![](https://img.shields.io/badge/platform-Linux-black?logo=linux&logoColor=white) ![](https://img.shields.io/badge/platform-MacOS-black?logo=apple&logoColor=white)

## CatBoost-Cgo

Evaluation library is the fastest way for inference a model CatBoost. The library provides a [C API](https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/c_api.h).\
The [C API](https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/c_api.h) interface can be accessed from any programming language (example Golang + [Cgo](https://go.dev/wiki/cgo)).

Prebuilt shared library (`*.so` | `*.dylib`) artifacts are available of the [releases](https://github.com/catboost/catboost/releases) page on GitHub CatBoost project.\
The shared library:

1) Should be in `/usr/local/lib`
2) Or set path in environment `CATBOOST_LIBRARY_PATH`
3) Or set path manual in source code `SetSharedLibraryPath` (see example below)

For more information, see <https://catboost.ai/en/docs/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper>.

## Compatibility

<table>
  <tr>
    <th>Previous versions</th>
    <th>v1.2.2</th>
    <th>v1.2.3</th>
    <th>v1.2.4</th>
    <th>v1.2.5</th>
    <th>v1.2.6</th>
    <th>v1.2.7</th>
    <th>v1.2.8</th>
  </tr>
  <tr>
    <td align="center">🚫 (not testing)</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
</table>

## Features

**Supported functionality** (<https://catboost.ai/en/docs/concepts/python-quickstart>):

+ CatBoostRegressor ✅
+ CatBoostClassifier ✅
+ CatBoostRanker ✅

**Supported prediction types** (<https://github.com/catboost/catboost/blob/master/catboost/libs/model/enums.h>):

+ RawFormulaVal ✅
+ Probability ✅
+ Class ✅
+ RMSEWithUncertainty ✅
+ Exponent ✅

## Limitation

**Supported operating system and architectures:**
<table>
  <tr>
    <th>Operating system</th>
    <th>CPU architectures</th>
    <th>GPU support using CUDA</th>
  </tr>
  <tr>
    <td>MacOS</td>
    <td >✅ (x86_64)</td>
    <td>🚫</td>
  </tr>
  <tr>
    <td>Linux</td>
    <td>✅ (x86_64)</td>
    <td>✅ (x86_64)</td>
  </tr>
  <tr>
    <td>Windows 10 and 11</td>
    <td>🚫</td>
    <td>🚫</td>
  </tr>
</table>

>_If you use GPU - only device 0 is supported for now. More details: <https://github.com/catboost/catboost/issues/2774>._

**Supported Type:**

+ Numeric ✅
+ Categorical ✅ (<https://catboost.ai/en/docs/features/categorical-features>)
+ Text 🚫 (<https://catboost.ai/en/docs/features/text-features>)
+ Embeddings 🚫 (<https://catboost.ai/en/docs/features/embeddings-features>)

## Installation

1) Install **[catboost-cgo](https://github.com/mirecl/catboost-cgo)**:

```go
go get github.com/mirecl/catboost-cgo
```

2) Download CatBoost shared library from release page: <https://github.com/catboost/catboost/releases>

3) Save CatBoost shared library in  `/usr/local/lib` or manual set path:

```go
import (
 cb "github.com/mirecl/catboost-cgo/catboost"
)

func main(){
  cb.SetSharedLibraryPath(...)
}
```

4) See [examples](example) of use

### Usage

+ [Regression](example/regressor)
+ [Binary classification](example/classifier)
+ [Multiclassification](example/multiclassification)
+ [Ranker](example/ranker)
+ [Titanic](example/titanic)
+ [Metadata](example/metadata)
+ [Uncertainty](example/uncertainty)
+ [Survival](example/survival)

### Thanks

+ [@lukangping](https://github.com/lukangping) for <https://github.com/lukangping/catboost-go>
+ [@bourbaki](https://github.com/bourbaki) for <https://github.com/bourbaki/catboost-go>
+ [@yalue](https://github.com/yalue) for <https://github.com/yalue/onnxruntime_go>
