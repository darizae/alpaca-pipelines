PYTHON ?= python3.11
VENV_PYTHON = ./.venv/bin/python

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
        env-init env-check runtime-check \
        create-training-run create-prediction-run create-evaluation-run \
        execute-run list-runs inspect-run cancel-run \
        generate-slurm submit-run \
        export-prediction-selection-tables clean-stale-workflow-job delete-failed-workflow-job \
        prediction-review-preview prediction-review-generate prediction-review-concat prediction-review-export \
        clean-run

venv:
	$(PYTHON) -m venv .venv

install:
	$(VENV_PYTHON) -m pip install -e ".[dev]" && ./.venv/bin/pre-commit install

lint:
	$(VENV_PYTHON) -m ruff check .

format:
	$(VENV_PYTHON) -m ruff format .

typecheck:
	$(VENV_PYTHON) -m mypy .

test:
	PYTHONPATH=src $(VENV_PYTHON) -m pytest -q

env-init:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

env-check:
	@test -f .env || (echo "Missing .env. Run: make env-init"; exit 1)
	@test -n "$(ALPACA_COLLECTION_ROOT)" || (echo "ALPACA_COLLECTION_ROOT missing in .env"; exit 1)
	@test -n "$(ALPACA_MERGED_INDEX)" || (echo "ALPACA_MERGED_INDEX missing in .env"; exit 1)
	@test -n "$(ALPACA_DATASETS_ROOT)" || (echo "ALPACA_DATASETS_ROOT missing in .env"; exit 1)
	@test -n "$(ALPACA_RUNS_ROOT)" || (echo "ALPACA_RUNS_ROOT missing in .env"; exit 1)

runtime-check: env-check
	@test -x "$(VENV_PYTHON)" || (echo "Missing virtualenv python at $(VENV_PYTHON). Run: make venv install"; exit 1)
	@test -x ./.venv/bin/alpaca-pipelines || (echo "Missing CLI entrypoint at ./.venv/bin/alpaca-pipelines. Run: make install"; exit 1)
	@repo_realpath="$$(realpath .)"; \
	module_path="$$(ALPACA_COLLECTION_ROOT="$(ALPACA_COLLECTION_ROOT)" ALPACA_MERGED_INDEX="$(ALPACA_MERGED_INDEX)" ALPACA_DATASETS_ROOT="$(ALPACA_DATASETS_ROOT)" ALPACA_RUNS_ROOT="$(ALPACA_RUNS_ROOT)" $(VENV_PYTHON) -c 'import pathlib; import alpaca_pipelines; print(pathlib.Path(alpaca_pipelines.__file__).resolve())')"; \
	case "$$module_path" in "$$repo_realpath"/*) ;; *) echo "alpaca_pipelines imports from $$module_path, not $$repo_realpath"; exit 1;; esac; \
	shebang_target="$$(head -n 1 ./.venv/bin/alpaca-pipelines | sed 's/^#!//')"; \
	case "$$shebang_target" in "$$repo_realpath"/*) ;; *) echo "CLI shebang points to $$shebang_target, not $$repo_realpath"; exit 1;; esac; \
	ALPACA_COLLECTION_ROOT="$(ALPACA_COLLECTION_ROOT)" ALPACA_MERGED_INDEX="$(ALPACA_MERGED_INDEX)" ALPACA_DATASETS_ROOT="$(ALPACA_DATASETS_ROOT)" ALPACA_RUNS_ROOT="$(ALPACA_RUNS_ROOT)" $(VENV_PYTHON) -m alpaca_pipelines.cli dataset-status --json >/dev/null; \
	echo "Runtime OK: $$module_path"

# --- Run lifecycle ---

create-training-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-training-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli create training --config "$(RUN_CONFIG)"

create-rf-training-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-rf-training-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli create rf_training --config "$(RUN_CONFIG)"

create-prediction-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-prediction-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli create prediction --config "$(RUN_CONFIG)"

create-evaluation-run: env-check
	@test -n "$(RUN_CONFIG)" || (echo "Usage: make create-evaluation-run RUN_CONFIG=<path>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli create evaluation --config "$(RUN_CONFIG)"

RUN_ID ?=
execute-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make execute-run RUN_ID=<id>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli execute --run-id "$(RUN_ID)"

list-runs: env-check
	@$(VENV_PYTHON) -m alpaca_pipelines.cli list

inspect-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make inspect-run RUN_ID=<id>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli inspect --run-id "$(RUN_ID)"

cancel-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make cancel-run RUN_ID=<id>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli cancel --run-id "$(RUN_ID)"

# --- Post-processing ---

USE_RF_FILTERED ?= 0
FREQ_LOW_HZ ?= 0
FREQ_HIGH_HZ ?= 4000

export-prediction-selection-tables: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make export-prediction-selection-tables RUN_ID=<id> [USE_RF_FILTERED=1] [FREQ_LOW_HZ=0] [FREQ_HIGH_HZ=4000]"; exit 1)
	@if [ "$(USE_RF_FILTERED)" = "1" ]; then \
		$(VENV_PYTHON) -m alpaca_pipelines.cli export-selection-tables --run-id "$(RUN_ID)" --freq-low-hz "$(FREQ_LOW_HZ)" --freq-high-hz "$(FREQ_HIGH_HZ)" --use-rf-filtered; \
	else \
		$(VENV_PYTHON) -m alpaca_pipelines.cli export-selection-tables --run-id "$(RUN_ID)" --freq-low-hz "$(FREQ_LOW_HZ)" --freq-high-hz "$(FREQ_HIGH_HZ)"; \
	fi

PREDICTION_REVIEW_MANIFEST ?=
PREDICTION_REVIEW_ITEM_ID ?=
PREDICTION_REVIEW_SPECTROGRAM_CONFIG ?=
PREDICTION_REVIEW_CONCAT_OUTPUT ?=
PREDICTION_REVIEW_EXPORT_DIR ?=

prediction-review-preview: env-check
	@test -n "$(PREDICTION_REVIEW_MANIFEST)" || (echo "Usage: make prediction-review-preview PREDICTION_REVIEW_MANIFEST=<path> PREDICTION_REVIEW_ITEM_ID=<id> [PREDICTION_REVIEW_SPECTROGRAM_CONFIG=<path>]"; exit 1)
	@test -n "$(PREDICTION_REVIEW_ITEM_ID)" || (echo "Usage: make prediction-review-preview PREDICTION_REVIEW_MANIFEST=<path> PREDICTION_REVIEW_ITEM_ID=<id> [PREDICTION_REVIEW_SPECTROGRAM_CONFIG=<path>]"; exit 1)
	@if [ -n "$(PREDICTION_REVIEW_SPECTROGRAM_CONFIG)" ]; then \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-preview --manifest "$(PREDICTION_REVIEW_MANIFEST)" --item-id "$(PREDICTION_REVIEW_ITEM_ID)" --spectrogram-config "$(PREDICTION_REVIEW_SPECTROGRAM_CONFIG)"; \
	else \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-preview --manifest "$(PREDICTION_REVIEW_MANIFEST)" --item-id "$(PREDICTION_REVIEW_ITEM_ID)"; \
	fi

prediction-review-generate: env-check
	@test -n "$(PREDICTION_REVIEW_MANIFEST)" || (echo "Usage: make prediction-review-generate PREDICTION_REVIEW_MANIFEST=<path> [PREDICTION_REVIEW_SPECTROGRAM_CONFIG=<path>]"; exit 1)
	@if [ -n "$(PREDICTION_REVIEW_SPECTROGRAM_CONFIG)" ]; then \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-generate --manifest "$(PREDICTION_REVIEW_MANIFEST)" --spectrogram-config "$(PREDICTION_REVIEW_SPECTROGRAM_CONFIG)"; \
	else \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-generate --manifest "$(PREDICTION_REVIEW_MANIFEST)"; \
	fi

prediction-review-concat: env-check
	@test -n "$(PREDICTION_REVIEW_MANIFEST)" || (echo "Usage: make prediction-review-concat PREDICTION_REVIEW_MANIFEST=<path> [PREDICTION_REVIEW_CONCAT_OUTPUT=<path>]"; exit 1)
	@if [ -n "$(PREDICTION_REVIEW_CONCAT_OUTPUT)" ]; then \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-concat --manifest "$(PREDICTION_REVIEW_MANIFEST)" --output-wav "$(PREDICTION_REVIEW_CONCAT_OUTPUT)"; \
	else \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-concat --manifest "$(PREDICTION_REVIEW_MANIFEST)"; \
	fi

prediction-review-export: env-check
	@test -n "$(PREDICTION_REVIEW_MANIFEST)" || (echo "Usage: make prediction-review-export PREDICTION_REVIEW_MANIFEST=<path> PREDICTION_REVIEW_EXPORT_DIR=<path> [PREDICTION_REVIEW_ITEM_ID=<id>]"; exit 1)
	@test -n "$(PREDICTION_REVIEW_EXPORT_DIR)" || (echo "Usage: make prediction-review-export PREDICTION_REVIEW_MANIFEST=<path> PREDICTION_REVIEW_EXPORT_DIR=<path> [PREDICTION_REVIEW_ITEM_ID=<id>]"; exit 1)
	@if [ -n "$(PREDICTION_REVIEW_ITEM_ID)" ]; then \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-export --manifest "$(PREDICTION_REVIEW_MANIFEST)" --destination-dir "$(PREDICTION_REVIEW_EXPORT_DIR)" --item-id "$(PREDICTION_REVIEW_ITEM_ID)"; \
	else \
		$(VENV_PYTHON) -m alpaca_pipelines.cli prediction-review-export --manifest "$(PREDICTION_REVIEW_MANIFEST)" --destination-dir "$(PREDICTION_REVIEW_EXPORT_DIR)"; \
	fi

clean-stale-workflow-job: env-check
	@test -n "$(JOB_ID)" || (echo "Usage: make clean-stale-workflow-job JOB_ID=<id> [ERROR_MESSAGE='...']"; exit 1)
	@error_message="$${ERROR_MESSAGE:-Marked failed by operator as stale: no worker process and no result artifact.}"; \
	$(VENV_PYTHON) -m alpaca_pipelines.cli fail-operation --job-id "$(JOB_ID)" --error-kind "StaleOperation" --error "$$error_message" --json

delete-failed-workflow-job: env-check
	@test -n "$(JOB_ID)" || (echo "Usage: make delete-failed-workflow-job JOB_ID=<id>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli delete-failed-operation --job-id "$(JOB_ID)" --json

# --- SLURM ---

generate-slurm: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make generate-slurm RUN_ID=<id>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli generate-slurm --run-id "$(RUN_ID)"

SLURM_CONFIG ?=
generate-slurm-with-config: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make generate-slurm-with-config RUN_ID=<id> SLURM_CONFIG=<path>"; exit 1)
	@test -n "$(SLURM_CONFIG)" || (echo "Usage: make generate-slurm-with-config RUN_ID=<id> SLURM_CONFIG=<path>"; exit 1)
	@$(VENV_PYTHON) -m alpaca_pipelines.cli generate-slurm --run-id "$(RUN_ID)" --slurm-config "$(SLURM_CONFIG)"

submit-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make submit-run RUN_ID=<id> [SLURM_CONFIG=<path>]"; exit 1)
	@if [ -n "$(SLURM_CONFIG)" ]; then \
		$(VENV_PYTHON) -m alpaca_pipelines.cli submit --run-id "$(RUN_ID)" --slurm-config "$(SLURM_CONFIG)"; \
	else \
		$(VENV_PYTHON) -m alpaca_pipelines.cli submit --run-id "$(RUN_ID)"; \
	fi

# --- Cleanup ---

clean-run: env-check
	@test -n "$(RUN_ID)" || (echo "Usage: make clean-run RUN_ID=<id>"; exit 1)
	@echo "This would remove run $(RUN_ID) under $(ALPACA_RUNS_ROOT). Use with care."
	@echo "Run manually: rm -rf $(ALPACA_RUNS_ROOT)/*/$(RUN_ID)"
