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



## Run tests
.PHONY: test
test:
	python -m pytest tests


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
