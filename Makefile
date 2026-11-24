# Sanket: one entry point for every reproducible step.
# Override PNPM if pnpm isn't on PATH, e.g.  make PNPM="npx -y pnpm@12.23.0" test

PNPM ?= pnpm
UV   ?= uv
ML   := cd ml &&
WEB  := cd web &&

.PHONY: help setup models inventory lint typecheck test test-ml test-web smoke e2e build ci clean

help:  ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

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

smoke: models  ## Phase 0 gate: pinned MediaPipe models run in Python AND in Chromium
	$(ML) $(UV) run pytest -m network
	$(WEB) $(PNPM) test:e2e

e2e: models  ## Playwright suite
	$(WEB) $(PNPM) test:e2e

build:  ## production build of the web app
	$(WEB) $(PNPM) build

ci: lint typecheck test smoke build  ## everything CI runs

clean:
	rm -rf web/.next web/test-results web/playwright-report ml/.pytest_cache ml/.mypy_cache ml/.ruff_cache
