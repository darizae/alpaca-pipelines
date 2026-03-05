

## 5.4 Prediction selection tables (post-processing output)

alpaca-pipelines supports post-processing a completed **prediction** run into
Raven-compatible selection tables (tab-separated). This is a persisted artifact
under the run directory.

### 5.4.1 Output location (required)

For a prediction run `<run_id>`:

```

ALPACA_RUNS_ROOT/
└── prediction/
└── <run_id>/
└── outputs/
└── predictions/
└── selection_tables/
├── selection_tables_summary.json
├── <audio_stem>.txt
└── <audio_stem>_rf_filtered.txt   (only when exporting RF-filtered tables)

```

Hard rules:
- All selection tables MUST be written under the run’s own directory.
- The system MUST NOT write outside `ALPACA_RUNS_ROOT`.
- The system MUST fail immediately if any expected prediction JSON file is missing.

### 5.4.2 Summary JSON contract

`selection_tables_summary.json` is a JSON object with EXACT fields:

- `generated_at`: string (ISO8601 Z)
- `predictions_dir`: string (absolute path)
- `selection_tables_dir`: string (absolute path)
- `use_rf_filtered`: bool
- `freq_low_hz`: int
- `freq_high_hz`: int
- `n_files`: int
- `files`: array of objects with EXACT fields:
  - `audio_file`: string (as recorded in prediction_summary.json)
  - `audio_file_stem`: string
  - `predictions_json`: string (absolute path to the per-file prediction json used)
  - `selection_table`: string (absolute path to the TSV written)

### 5.4.3 Raven selection table columns

Each TSV MUST include these columns (tab-separated), in this order:

- `Selection`
- `View`
- `Channel`
- `Begin Time (s)`
- `End Time (s)`
- `Low Freq (Hz)`
- `High Freq (Hz)`
- `Score`

Hard rule:
- Even if there are zero detections, the TSV MUST still include the header row.
