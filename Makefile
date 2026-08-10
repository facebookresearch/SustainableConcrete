# Local CI-equivalent targets. Mirrors the suite of GitHub Actions
# workflows in .github/workflows/ so that `make check-all` matches the
# checks a PR will run on CI.
#
#   make help               - list targets
#
# === Fast (seconds) ===
#   make lint               - black --check + flake8     (tests.yml :lint)
#   make format             - apply black formatting in-place
#   make test-py            - pytest with 100% coverage  (tests.yml :test)
#   make test-js            - JS GP sync test            (js-sync.yml)
#   make test-notebook-fmt  - nbformat validation        (notebooks.yml :notebook-lint)
#
# === Slow (minutes) ===
#   make test-notebooks     - execute every notebook     (notebooks.yml :mode-*)
#   make test-e2e           - Playwright desktop+mobile  (e2e.yml)
#   make test-lighthouse    - Lighthouse CI              (lighthouse.yml)
#
# === Aggregates ===
#   make test               - fast tests (py + js + notebook-fmt)
#   make check              - lint + test          (recommended pre-commit gate)
#   make check-all          - lint + every test    (full local CI parity)
#
# Most targets need extras installed:
#   pip install -e ".[dev]"        # lint, test-py
#   pip install -e ".[notebooks]"  # test-notebook-fmt, test-notebooks
#   npm ci                         # test-js, test-e2e, test-lighthouse
#   npx playwright install --with-deps chromium   # test-e2e

PYTHON ?= python

.PHONY: help \
        lint format \
        test-py test-js test-notebook-fmt test-notebooks \
        test-e2e test-lighthouse \
        test check check-all

help:
	@echo "Local CI-equivalent targets (see Makefile header for full list):"
	@echo ""
	@echo "  Fast:"
	@echo "    make lint               - black --check + flake8"
	@echo "    make format             - apply black formatting in-place"
	@echo "    make test-py            - pytest with 100% coverage gate"
	@echo "    make test-js            - JS GP sync test"
	@echo "    make test-notebook-fmt  - nbformat validation"
	@echo ""
	@echo "  Slow:"
	@echo "    make test-notebooks     - execute every notebook"
	@echo "    make test-e2e           - Playwright (desktop + mobile)"
	@echo "    make test-lighthouse    - Lighthouse CI"
	@echo ""
	@echo "  Aggregates:"
	@echo "    make test               - fast tests"
	@echo "    make check              - lint + test (recommended)"
	@echo "    make check-all          - lint + every test (full CI parity)"

# Tracked Python files. CI's `black --check --diff .` only sees files
# in the checked-out commit, so locally we must scope to tracked files
# too — otherwise an untracked work-in-progress script in the working
# tree would block `make lint` even though it can't fail the PR. Falls
# back from `sl files` (Sapling) to `git ls-files` (vanilla git).
TRACKED_PY := $(shell (sl files 2>/dev/null || git ls-files) | grep -E '\.py$$')

# --- Lint -----------------------------------------------------------
# Mirrors .github/workflows/tests.yml :lint, scoped to TRACKED_PY so
# untracked working-tree files don't poison the result. Black version
# is pinned via pyproject.toml's [project.optional-dependencies].dev
# so the formatter output is bit-for-bit identical to CI.
lint:
	$(PYTHON) -m black --check --diff $(TRACKED_PY)
	$(PYTHON) -m flake8 $(TRACKED_PY) --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 $(TRACKED_PY) --count --exit-zero --statistics

format:
	$(PYTHON) -m black $(TRACKED_PY)

# --- Python unit tests ---------------------------------------------
# Mirrors .github/workflows/tests.yml :test. We drop --cov-report=xml
# because we don't need the coverage.xml artefact locally.
test-py:
	$(PYTHON) -m pytest test/ -v --tb=short --cov=boxcrete --cov-report=term-missing --cov-fail-under=100

# --- JS GP sync test ------------------------------------------------
# Mirrors .github/workflows/js-sync.yml. Verifies docs/gp.mjs predicts
# the same values as the Python reference for the committed model.
# Every JS test CI runs (.github/workflows/js-sync.yml). Kept in lockstep
# with that workflow: `make test-js` green must imply the js-sync job is
# green, otherwise local runs give false confidence.
JS_TESTS = \
  test/test_js_gp.mjs \
  test/test_js_feature_parity.mjs \
  test/test_js_strength_v2.mjs \
  test/test_js_predictor_parity.mjs \
  test/test_js_physical_constraints.mjs \
  test/test_js_ui_smoke.mjs \
  test/test_js_units.mjs \
  test/test_lengthscales_v2.mjs \
  test/test_curve_monotonicity.mjs \
  test/test_data_freshness.mjs \
  test/test_js_preview_state.mjs \
  test/test_js_categorical_source.mjs

test-js:
	@for t in $(JS_TESTS); do \
	  printf '  -- %s --\n' "$$t"; \
	  node "$$t" > /dev/null || { echo "FAILED: $$t"; node "$$t"; exit 1; }; \
	done
	@echo "All $(words $(JS_TESTS)) JS tests passed."

# --- Notebook format validation ------------------------------------
# Mirrors .github/workflows/notebooks.yml :notebook-lint. Just validates
# the JSON; does not execute the notebooks.
test-notebook-fmt:
	$(PYTHON) -c "import nbformat, pathlib; \
nbs = sorted(pathlib.Path('notebooks').glob('*.ipynb')); \
[(nbformat.read(open(p), as_version=4), print(f'OK {p}')) for p in nbs]; \
print(f'Validated {len(nbs)} notebook(s).')"

# --- Notebook execution --------------------------------------------
# Mirrors .github/workflows/notebooks.yml :mode-{dependent,independent}.
# Slow: each notebook runs nbconvert with a 600s timeout.
# Mode-dependent notebooks are executed once per (mode, include-cost)
# combination, matching the CI matrix.
#
# Unlike CI (which writes --inplace and uploads the executed notebook as
# an artifact), we redirect output to a temp dir so the local working
# copy isn't dirtied by `make` runs.
NOTEBOOK_OUT := $(CURDIR)/.notebook-runs
NBCONVERT := $(PYTHON) -m jupyter nbconvert --to notebook --execute \
    --output-dir=$(NOTEBOOK_OUT) \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3
test-notebooks:
	@mkdir -p $(NOTEBOOK_OUT)
	BOXCRETE_OPTIMIZATION_MODE=concrete BOXCRETE_INCLUDE_COST=false \
	    $(NBCONVERT) --output=mode_concrete_cost_false.ipynb \
	    notebooks/prediction_and_optimization_tutorial.ipynb
	BOXCRETE_OPTIMIZATION_MODE=concrete BOXCRETE_INCLUDE_COST=true \
	    $(NBCONVERT) --output=mode_concrete_cost_true.ipynb \
	    notebooks/prediction_and_optimization_tutorial.ipynb
	BOXCRETE_OPTIMIZATION_MODE=mortar BOXCRETE_INCLUDE_COST=false \
	    $(NBCONVERT) --output=mode_mortar_cost_false.ipynb \
	    notebooks/prediction_and_optimization_tutorial.ipynb
	BOXCRETE_OPTIMIZATION_MODE=mortar BOXCRETE_INCLUDE_COST=true \
	    $(NBCONVERT) --output=mode_mortar_cost_true.ipynb \
	    notebooks/prediction_and_optimization_tutorial.ipynb
	@for nb in notebooks/*.ipynb; do \
	    case "$$nb" in \
	        notebooks/prediction_and_optimization_tutorial.ipynb) ;; \
	        *) echo "=== Executing $$nb ==="; \
	           $(NBCONVERT) "$$nb" ;; \
	    esac \
	done

# --- E2E (Playwright) ----------------------------------------------
# Mirrors .github/workflows/e2e.yml. Requires `npm ci` + a one-time
# `npx playwright install --with-deps chromium` to set up browsers.
test-e2e:
	npx playwright test --project=desktop
	npx playwright test --project=mobile

# --- Lighthouse CI -------------------------------------------------
# Mirrors .github/workflows/lighthouse.yml.
test-lighthouse:
	npx lhci autorun

# --- Aggregates ----------------------------------------------------
test: test-py test-js test-notebook-fmt
check: lint test
check-all: lint test-py test-js test-notebook-fmt test-notebooks test-e2e test-lighthouse
