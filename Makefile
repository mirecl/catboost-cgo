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