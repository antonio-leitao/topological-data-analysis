.PHONY: help install build-py build-rust test test-core test-python-dense test-python-sparse bench bench-save profile profile-build publish-py publish-rust clean
.DEFAULT_GOAL := help

# ── Colors ────────────────────────────────────────────────────────────────
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m

# ── Config ────────────────────────────────────────────────────────────────
BENCH_CRATE := tda_core
BENCH_NAME  := persistent_homology
PYTHON      := crates/python/.venv/bin/python
MATURIN     := crates/python/.venv/bin/maturin
BENCH_FILTER ?=
MAX_DIM ?= 1
PROFILE_TIME ?= 20
PROFILE_MAX_DIM ?= 2
PROFILE_FILTER ?= sparse_h2
PROFILE_OUT ?= crates/core/profile.json.gz
PROFILE_RATE ?= 1000
SAMPLY ?= samply

help: ## Show this help message
	@echo "$(CYAN)Available commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-14s$(RESET) %s\n", $$1, $$2}'

# ── Build / install ───────────────────────────────────────────────────────

install: ## Build & install the Python bindings in editable mode (release)
	@echo "$(CYAN)📦 Installing Python bindings (dev)...$(RESET)"
	@cd crates/python && env -u CONDA_PREFIX VIRTUAL_ENV=$$PWD/.venv PATH=$$PWD/.venv/bin:$$PATH .venv/bin/maturin develop --release --skip-install

build-py: ## Build a Python wheel under target/wheels
	@echo "$(CYAN)📦 Building Python wheel...$(RESET)"
	@cd crates/python && uv run maturin build --release

build-rust: ## Build the Rust core crate in release mode
	@echo "$(CYAN)📦 Building Rust core crate...$(RESET)"
	@cargo build --release -p $(BENCH_CRATE)

# ── Test ──────────────────────────────────────────────────────────────────

test-core: ## Run Rust core tests
	@echo "$(CYAN)🧪 Running Rust core tests...$(RESET)"
	@cargo test -p $(BENCH_CRATE)

test-python-dense: install ## Run Python dense correctness test
	@echo "$(CYAN)🧪 Running Python dense correctness...$(RESET)"
	@$(PYTHON) crates/python/tests/correctness.py dense $(MAX_DIM)

test-python-sparse: install ## Run Python sparse correctness test
	@echo "$(CYAN)🧪 Running Python sparse correctness...$(RESET)"
	@$(PYTHON) crates/python/tests/correctness.py sparse $(MAX_DIM)

test: test-core test-python-dense test-python-sparse ## Run all correctness tests
	@echo "$(GREEN)✅ Tests passed!$(RESET)"

# ── Bench ─────────────────────────────────────────────────────────────────

bench-save: ## Run benchmarks, save baseline as current branch, compare against previous run of same branch
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	echo "$(CYAN)📊 Running benchmarks, saving baseline '$$BRANCH'...$(RESET)"; \
	TDA_MAX_DIM=$(MAX_DIM) cargo bench -p $(BENCH_CRATE) --bench $(BENCH_NAME) -- --save-baseline $$BRANCH $(BENCH_FILTER); \
	echo "$(GREEN)✅ Baseline '$$BRANCH' saved.$(RESET)"

bench: ## Compare current code against a chosen saved baseline (does not save)
	@BASELINES=$$(find target/criterion -type f -name estimates.json 2>/dev/null \
		| xargs -n1 dirname 2>/dev/null \
		| xargs -n1 basename 2>/dev/null \
		| sort -u \
		| grep -vE '^(new|change|base|report)$$'); \
	if [ -z "$$BASELINES" ]; then \
		echo "$(RED)No saved baselines found. Run 'make bench-save' first.$(RESET)"; \
		exit 1; \
	fi; \
	BASELINE=$$(echo "$$BASELINES" | gum choose --header "Select baseline to compare against:"); \
	if [ -z "$$BASELINE" ]; then exit 0; fi; \
	echo "$(CYAN)📊 Comparing against baseline '$$BASELINE'...$(RESET)"; \
	TDA_MAX_DIM=$(MAX_DIM) cargo bench -p $(BENCH_CRATE) --bench $(BENCH_NAME) -- --baseline $$BASELINE $(BENCH_FILTER)

profile-build: ## Build the Criterion bench binary for profiling
	@echo "$(CYAN)📦 Building benchmark binary for profiling...$(RESET)"
	@TDA_MAX_DIM=$(PROFILE_MAX_DIM) cargo bench -p $(BENCH_CRATE) --bench $(BENCH_NAME) --no-run

profile: profile-build ## Profile the sparse H2 Criterion benchmark group with samply
	@mkdir -p "$$(dirname "$(PROFILE_OUT)")"
	@BENCH_BIN=$$(find target/release/deps -maxdepth 1 -type f -perm -111 -name '$(BENCH_NAME)-*' | sort | tail -n 1); \
	if [ -z "$$BENCH_BIN" ]; then \
		echo "$(RED)Could not find target/release/deps/$(BENCH_NAME)-* bench executable.$(RESET)"; \
		exit 1; \
	fi; \
	echo "$(CYAN)📊 Profiling $(PROFILE_FILTER) via $$BENCH_BIN...$(RESET)"; \
	TDA_MAX_DIM=$(PROFILE_MAX_DIM) $(SAMPLY) record \
		--save-only \
		--unstable-presymbolicate \
		--main-thread-only \
		--rate $(PROFILE_RATE) \
		-o "$(PROFILE_OUT)" \
		-- "$$BENCH_BIN" "$(PROFILE_FILTER)" --profile-time $(PROFILE_TIME) --noplot; \
	echo "$(GREEN)✅ Profile written to $(PROFILE_OUT).$(RESET)"

# ── Publish ───────────────────────────────────────────────────────────────

release-py: ## Tag a Python release (e.g. make release-py VERSION=0.2.2)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Usage: make release-py VERSION=x.y.z$(RESET)"; exit 1; \
	fi
	@echo "$(YELLOW)🏷  Tagging v$(VERSION) and pushing...$(RESET)"
	@git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@git push origin "v$(VERSION)"
	@echo "$(GREEN)✅ Tag pushed. CI will build & upload to PyPI.$(RESET)"

publish-rust: ## Publish the Rust core crate to crates.io
	@echo "$(YELLOW)🚀 Publishing Rust core crate to crates.io...$(RESET)"
	@cargo publish -p $(BENCH_CRATE)

# ── Misc ──────────────────────────────────────────────────────────────────

clean: ## Remove build artefacts (target/, wheels)
	@echo "$(CYAN)🧹 Cleaning...$(RESET)"
	@cargo clean
	@rm -rf target/wheels
