# Sanket: one entry point for every reproducible step.
# Override PNPM if pnpm isn't on PATH, e.g.  make PNPM="npx -y pnpm@12.23.0" test

PNPM ?= pnpm
UV   ?= uv
ML   := cd ml &&
WEB  := cd web &&

.PHONY: help setup models inventory lint typecheck test test-ml test-web smoke e2e build ci clean \
        samples check-video-metadata check-extractor bench-page extract upload \
        parity-fixture check-fps check-downscale setup-train typecheck-train test-train \
        record-page import-recordings extract-include50

help:  ## list targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

setup:  ## install pinned Python (uv) and web (pnpm) environments
	$(ML) $(UV) sync --locked
	$(WEB) $(PNPM) install --frozen-lockfile

models:  ## fetch the pinned MediaPipe .task files (sha256-checked) for Python and the web app
	$(ML) $(UV) run python -m sanket.spec --copy-to-web

inventory:  ## regenerate reports/phase0_inventory.{md,json} and ml/data/manifests from Zenodo metadata
	$(ML) $(UV) run python -m sanket.data.inventory

lint:  ## ruff + eslint
	$(ML) $(UV) run ruff check src tests
	$(ML) $(UV) run ruff format --check src tests
	$(WEB) $(PNPM) lint

# Torch code is type-checked in the training env (typecheck-train); the main env has no torch.
TORCH_EXCLUDE := --exclude 'src/sanket/(train\.py|evaluate\.py|models/)' --exclude 'tests/test_train\.py'
TRAIN_ENV := UV_PROJECT_ENVIRONMENT=.venv-train

typecheck:  ## mypy (strict) + tsc (torch modules: see typecheck-train)
	$(ML) $(UV) run mypy $(TORCH_EXCLUDE)
	$(WEB) $(PNPM) typecheck

setup-train:  ## separate training env (torch, tensorboard) in ml/.venv-train
	$(ML) $(TRAIN_ENV) $(UV) sync --locked --group train

typecheck-train:  ## mypy (strict) over everything, in the training env
	$(ML) $(TRAIN_ENV) $(UV) run --no-sync mypy

test-train:  ## GRU + training-loop tests on synthetic data (training env, low priority)
	$(ML) $(TRAIN_ENV) nice -n 19 $(UV) run --no-sync pytest -q tests/test_train.py

test: test-ml test-web  ## offline unit tests (Python + web)

test-ml:
	$(ML) $(UV) run pytest -m "not network"

test-web:
	$(WEB) $(PNPM) test

smoke: models parity-fixture build  ## Phase 0 gate: pinned MediaPipe models run in Python AND in Chromium
	$(ML) $(UV) run pytest -m network
	$(WEB) $(PNPM) test:e2e

e2e: models parity-fixture build  ## Playwright suite (against the production build)
	$(WEB) $(PNPM) test:e2e

build:  ## production build of the web app
	$(WEB) $(PNPM) build

ci: lint typecheck test smoke  ## everything CI runs (smoke includes the production build)

# ---- Phase 1 pre-checks (ADR 002, ADR 005, video decoding) ----

samples:  ## pick the seeded check samples and fetch those ~80 videos (~1.1 GB, Range requests)
	$(ML) $(UV) run python -m sanket.checks.sample --fetch

check-video-metadata: samples  ## OpenCV vs ffprobe on 30 .MOV + the .MP4 files (needs ffprobe)
	$(ML) $(UV) run python -m sanket.checks.video_metadata

check-extractor: samples models  ## ADR 002 measure (a): Holistic vs Hand+Pose on 50 videos
	$(ML) $(UV) run python -m sanket.checks.extractor_compare

parity-fixture: models  ## decode an INCLUDE-50 test video + Python reference for the browser parity tests
	$(ML) $(UV) run python -m sanket.checks.browser_parity_fixture

check-fps: models  ## evidence for ADR 002's fps rule (needs: make samples)
	$(ML) $(UV) run python -m sanket.checks.fps_sensitivity --extractor holistic

check-downscale: parity-fixture  ## ADR 007: browser downscale methods vs OpenCV INTER_AREA (Chromium + Chrome)
	$(WEB) node scripts/downscale-probe.mjs

record-page: models  ## Phase 6: serve the keypoints-only recorder at /dev/record (see tools/record/)
	$(WEB) $(PNPM) build
	@echo "Open http://localhost:3000/dev/record in Chrome"
	$(WEB) $(PNPM) exec next start -p 3000

import-recordings:  ## validate + convert recorder session files: make import-recordings FILES="a.json b.json"
	@test -n "$(FILES)" || (echo 'set FILES="path/to/sanket-rec-*.json"'; exit 1)
	$(ML) $(UV) run python -m sanket.data.recorded $(FILES)

bench-page: models  ## ADR 002 measure (b): serve the browser benchmark at /dev/extractor-bench
	$(WEB) $(PNPM) build
	@echo "Open http://localhost:3000/dev/extractor-bench in Chrome (plugged in, tab in front)"
	$(WEB) $(PNPM) exec next start -p 3000

# ---- Phase 1 extraction (starts only after ADR 002 is decided; see ml/configs/extract.yaml) ----

extract: models  ## download + extract all 44 zips, one at a time (resumable; aborts below 10 GB free)
	$(ML) $(UV) run python -m sanket.pipeline.extract

extract-include50: models  ## only the 958 INCLUDE-50 videos, pulled per video by HTTP Range (~13 GB, no full zips)
	$(ML) $(UV) run python -m sanket.pipeline.extract --only include50

upload:  ## upload keypoints as a PRIVATE Kaggle dataset: make upload KAGGLE_DATASET=user/slug [ONLY=include50]
	@test -n "$(KAGGLE_DATASET)" || (echo "set KAGGLE_DATASET=<kaggle-user>/<slug>"; exit 1)
	$(ML) $(UV) run --group upload python -m sanket.pipeline.upload --dataset $(KAGGLE_DATASET) $(if $(ONLY),--only $(ONLY))

clean:
	rm -rf web/.next web/test-results web/playwright-report ml/.pytest_cache ml/.mypy_cache ml/.ruff_cache
