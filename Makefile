train:
	@echo "Classifier"
	@python example/classifier/classifier.py 
	@echo ""
	@echo "Regressor"
	@python example/regressor/regressor.py 
	@echo ""
	@echo "Multiclassification"
	@python example/multiclassification/multiclassification.py 
	@echo ""
	@echo "Metadata"
	@python example/metadata/metadata.py 
	@echo ""
	@echo "Uncertainty"
	@python example/uncertainty/uncertainty.py 
	@echo ""
	@echo "Survival"
	@python example/survival/survival.py 

predict:
	@echo "Classifier"
	@go run example/classifier/classifier.go
	@echo ""
	@echo "Regressor"
	@go run example/regressor/regressor.go
	@echo ""
	@echo "Multiclassification"
	@go run example/multiclassification/multiclassification.go
	@echo ""
	@echo "Metadata"
	@go run example/metadata/metadata.go
	@echo ""
	@echo "Uncertainty"
	@go run example/uncertainty/uncertainty.go
	@echo ""
	@echo "Survival"
	@go run example/survival/survival.go