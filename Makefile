#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = caso_berka_project
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python
ifneq (,$(wildcard .venv/bin/python))
PYTHON_INTERPRETER = .venv/bin/python
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run unit tests
.PHONY: test-unit
test-unit:
	$(PYTHON_INTERPRETER) -m pytest tests/unit -m unit

## Run integration tests
.PHONY: test-integration
test-integration:
	$(PYTHON_INTERPRETER) -m pytest tests/integration -m integration

## Run all tests (excluding slow e2e)
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests -m "not slow"

## Run all tests including slow e2e
.PHONY: test-all
test-all:
	$(PYTHON_INTERPRETER) -m pytest tests

## Lint and test (CI local)
.PHONY: ci-local
ci-local: lint test-unit test-integration


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m caso_berka_model.dataset

## Train models
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m caso_berka_model.modeling.train

## Reproduce the DVC pipeline
.PHONY: dvc-repro
dvc-repro:
	PATH="$(CURDIR)/.venv/bin:$(PATH)" dvc repro

## Pull data from the local DVC remote
.PHONY: dvc-pull
dvc-pull:
	PATH="$(CURDIR)/.venv/bin:$(PATH)" dvc pull

## Push data to the local DVC remote
.PHONY: dvc-push
dvc-push:
	PATH="$(CURDIR)/.venv/bin:$(PATH)" dvc push

## Train models and log the run to local MLflow (SQLite)
.PHONY: mlflow-train
mlflow-train:
	$(PYTHON_INTERPRETER) -m caso_berka_model.mlflow_engine.run

## Open the local MLflow UI (http://127.0.0.1:5000)
.PHONY: mlflow-ui
mlflow-ui:
	$(PYTHON_INTERPRETER) -m mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlflow.db --host 127.0.0.1 --port 5000

## Serve the Production model from the local Model Registry
.PHONY: mlflow-serve
mlflow-serve:
	MLFLOW_TRACKING_URI=sqlite:///$(CURDIR)/mlflow.db $(PYTHON_INTERPRETER) -m mlflow models serve --model-uri models:/Berka_BuenCliente@Production --host 0.0.0.0 --port 8080 --env-manager local

## Serve the custom FastAPI prediction API (Production PyFunc)
.PHONY: api-serve
api-serve:
	MLFLOW_TRACKING_URI=sqlite:///$(CURDIR)/mlflow.db $(PYTHON_INTERPRETER) -m uvicorn caso_berka_model.api.main:app --host 0.0.0.0 --port 8000

## Copy Production PyFunc artifacts to models/docker_production for the Docker image
.PHONY: docker-prepare-model
docker-prepare-model:
	$(PYTHON_INTERPRETER) -m caso_berka_model.api.prepare_docker_model

## Build the API Docker image (prepares Production model first)
.PHONY: docker-build
docker-build: docker-prepare-model
	docker build -t caso-berka-api .

## Run the API container on port 8000
.PHONY: docker-run
docker-run:
	docker run --rm -p 8000:8000 \
		--env-file models/docker_meta.env \
		-e ENVIRONMENT=docker \
		caso-berka-api


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
