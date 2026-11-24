# Sanket: one entry point for every reproducible step.
# Override PNPM if pnpm isn't on PATH, e.g.  make PNPM="npx -y pnpm@12.23.0" test

PNPM ?= pnpm
UV   ?= uv
ML   := cd ml &&
WEB  := cd web &&

.PHONY: help setup models inventory lint typecheck test test-ml test-web smoke e2e build ci clean \
        samples check-video-metadata check-extractor bench-page extract upload

help:  ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

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

typecheck:  ## mypy (strict) + tsc
	$(ML) $(UV) run mypy
	$(WEB) $(PNPM) typecheck

test: test-ml test-web  ## offline unit tests (Python + web)

test-ml:
	$(ML) $(UV) run pytest -m "not network"

test-web:
	$(WEB) $(PNPM) test

smoke: models build  ## Phase 0 gate: pinned MediaPipe models run in Python AND in Chromium
	$(ML) $(UV) run pytest -m network
	$(WEB) $(PNPM) test:e2e

e2e: models build  ## Playwright suite (against the production build)
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

bench-page: models  ## ADR 002 measure (b): serve the browser benchmark at /dev/extractor-bench
	$(WEB) $(PNPM) build
	@echo "Open http://localhost:3000/dev/extractor-bench in Chrome (plugged in, tab in front)"
	$(WEB) $(PNPM) exec next start -p 3000

# ---- Phase 1 extraction (starts only after ADR 002 is decided; see ml/configs/extract.yaml) ----

extract: models  ## download + extract all 44 zips, one at a time (resumable; aborts below 10 GB free)
	$(ML) $(UV) run python -m sanket.pipeline.extract

upload:  ## upload keypoints as a PRIVATE Kaggle dataset: make upload KAGGLE_DATASET=user/slug
	@test -n "$(KAGGLE_DATASET)" || (echo "set KAGGLE_DATASET=<kaggle-user>/<slug>"; exit 1)
	$(ML) $(UV) run --group upload python -m sanket.pipeline.upload --dataset $(KAGGLE_DATASET)

clean:
	rm -rf web/.next web/test-results web/playwright-report ml/.pytest_cache ml/.mypy_cache ml/.ruff_cache
