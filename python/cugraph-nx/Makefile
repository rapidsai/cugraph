# Copyright (c) 2023, NVIDIA CORPORATION.
SHELL= /bin/bash

lint:
	git ls-files | xargs pre-commit run --config lint.yaml --files
lint-update:
	pre-commit autoupdate --config lint.yaml
