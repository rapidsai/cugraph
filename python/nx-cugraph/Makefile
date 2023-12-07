# Copyright (c) 2023, NVIDIA CORPORATION.
SHELL= /bin/bash

.PHONY: all
all: plugin-info lint

.PHONY: lint
lint:
	git ls-files | xargs pre-commit run --config lint.yaml --files

.PHONY: lint-update
lint-update:
	pre-commit autoupdate --config lint.yaml

.PHONY: plugin-info
plugin-info:
	python _nx_cugraph/__init__.py
