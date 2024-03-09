# Copyright (c) 2023-2024, NVIDIA CORPORATION.
SHELL= /bin/bash

.PHONY: all
all: plugin-info lint readme

.PHONY: lint
lint:
	git ls-files | xargs pre-commit run --config lint.yaml --files || true

.PHONY: lint-update
lint-update:
	pre-commit autoupdate --config lint.yaml

.PHONY: plugin-info
plugin-info:
	python _nx_cugraph/__init__.py

objects.inv:
	wget https://networkx.org/documentation/stable/objects.inv

.PHONY: readme
readme: objects.inv
	python scripts/update_readme.py README.md objects.inv
