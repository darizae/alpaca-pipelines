PYTHON ?= python3.11
VENV_ACTIVATE = . .venv/bin/activate

ENV_FILE := .env
ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
export
endif

ALPACA_COLLECTION_ROOT ?=
ALPACA_MERGED_INDEX ?=
ALPACA_DATASETS_ROOT ?=
ALPACA_RUNS_ROOT ?=

RUN_CONFIG ?=

.PHONY: venv install lint format typecheck test \
        env-init env-check \
        create-training-run create-prediction-run create-evaluation-run \
        execute-run list-runs inspect-run cancel-run \
        generate-slurm submit-slurm \
        export-prediction-selection-tables \
        clean-run

venv:
	$(PYTHON) -m venv .venv

install:
	$(VENV_ACTIVATE) && pip install -e ".[dev]" && pre-commit install

lint:
	$(VENV_ACTIVATE) && ruff check .

format:
	$(VENV_ACTIVATE) && ruff format .

typecheck:
	$(VENV_ACTIVATE) && mypy .

test:
	$(VENV_ACTIVATE) && pytest -q

env-init:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

env-check:
	@test -f .env || (echo "Missing .env. Run: make env-init"; exit 1)
	@test -n "$(ALPACA_COLLECTION_ROOT)" || (echo "ALPACA_COLLECTION_ROOT missing in .env"; exit 1)
	@test -n "$(ALPACA_MERGED_INDEX)" || (echo "ALPACA_MERGED_INDEX missing in .env"; exit 1)
	@test -n "$(ALPACA_DATASETS_ROOT)" || (echo "ALPACA_DATASETS_ROOT missing in .env"; exit 1)
	@test -n "$(ALPACA_RUNS_ROOT)" || (echo "ALPACA_RUNS_ROOT missing in .env"; exit 1)

# --- Run lifecycle ---

create-training-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-training-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines create training --config "$(RUN_CONFIG)"

create-rf-training-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-rf-training-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines create rf_training --config "$(RUN_CONFIG)"

create-prediction-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-prediction-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines create prediction --config "$(RUN_CONFIG)"

create-evaluation-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-evaluation-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines create evaluation --config "$(RUN_CONFIG)"

RUN_ID ?=
execute-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make execute-run RUN_ID=<id>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines execute --run-id "$(RUN_ID)"

list-runs: env-check
	@$(VENV_ACTIVATE) && alpaca-pipelines list

inspect-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make inspect-run RUN_ID=<id>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines inspect --run-id "$(RUN_ID)"

cancel-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make cancel-run RUN_ID=<id>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines cancel --run-id "$(RUN_ID)"

# --- Post-processing ---

USE_RF_FILTERED ?= 0
FREQ_LOW_HZ ?= 0
FREQ_HIGH_HZ ?= 4000

export-prediction-selection-tables: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make export-prediction-selection-tables RUN_ID=<id> [USE_RF_FILTERED=1] [FREQ_LOW_HZ=0] [FREQ_HIGH_HZ=4000]"; exit 1)
	@if [ "$(USE_RF_FILTERED)" = "1" ]; then \
		$(VENV_ACTIVATE) && alpaca-pipelines export-selection-tables --run-id "$(RUN_ID)" --freq-low-hz "$(FREQ_LOW_HZ)" --freq-high-hz "$(FREQ_HIGH_HZ)" --use-rf-filtered; \
	else \
		$(VENV_ACTIVATE) && alpaca-pipelines export-selection-tables --run-id "$(RUN_ID)" --freq-low-hz "$(FREQ_LOW_HZ)" --freq-high-hz "$(FREQ_HIGH_HZ)"; \
	fi

# --- SLURM ---

generate-slurm: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make generate-slurm RUN_ID=<id>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines generate-slurm --run-id "$(RUN_ID)"

SLURM_CONFIG ?=
generate-slurm-with-config: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make generate-slurm-with-config RUN_ID=<id> SLURM_CONFIG=<path>"; exit 1)
	@test -n "$(SLURM_CONFIG)" || (echo "Usage: make generate-slurm-with-config RUN_ID=<id> SLURM_CONFIG=<path>"; exit 1)
	@$(VENV_ACTIVATE) && alpaca-pipelines generate-slurm --run-id "$(RUN_ID)" --slurm-config "$(SLURM_CONFIG)"

# --- Cleanup ---

clean-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make clean-run RUN_ID=<id>"; exit 1)
	@echo "This would remove run $(RUN_ID) under $(ALPACA_RUNS_ROOT). Use with care."
	@echo "Run manually: rm -rf $(ALPACA_RUNS_ROOT)/*/$(RUN_ID)"
