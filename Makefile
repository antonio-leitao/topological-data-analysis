.PHONY: help install build-py build-rust test test-core test-python bench bench-peel bench-quot bench-clps profile profile-build profile-all release-py publish-rust clean
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
MAX_DIM ?= 2
VARIANT ?= vanilla
PROFILE_RATE ?= 1000
# Outside target/ so `cargo clean` keeps profiles (gitignored).
PROFILE_DIR ?= profiles
PROFILE_REPEATS ?= 1
PROFILE_PROFILE     ?= profiling
PROFILE_TARGET_DIR  ?= target/$(PROFILE_PROFILE)
DATASET ?=
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

test-python: install ## Run Python correctness test
	@echo "$(CYAN)🧪 Running Python correctness...$(RESET)"
	@$(PYTHON) crates/python/tests/correctness.py $(MAX_DIM)
	@echo "$(CYAN)🧪 Running coordinates checks...$(RESET)"
	@$(PYTHON) crates/python/tests/coordinates.py

test: test-core test-python ## Run all correctness tests
	@echo "$(GREEN)✅ Tests passed!$(RESET)"

# ── Bench ─────────────────────────────────────────────────────────────────

bench: ## Run BitCSR core benchmarks (VARIANT=vanilla|peel|quot)
	@echo "$(CYAN)📊 Running BitCSR benchmarks ($(VARIANT))...$(RESET)"
	@TDA_MAX_DIM=$(MAX_DIM) TDA_VARIANT=$(VARIANT) cargo bench -p $(BENCH_CRATE) --bench $(BENCH_NAME)

bench-peel: ## Benchmark the vanilla pipeline + peel
	@$(MAKE) bench VARIANT=peel

bench-quot: ## Benchmark the vanilla pipeline + quotient
	@$(MAKE) bench VARIANT=quot

bench-clps: ## Benchmark the vanilla pipeline + edge collapse
	@$(MAKE) bench VARIANT=clps

bench-all:
	@$(MAKE) bench
	@$(MAKE) bench VARIANT=peel
	@$(MAKE) bench VARIANT=quot
	@$(MAKE) bench VARIANT=clps



profile-build: ## Build the single-run profiling runner (optimized + line info)
	@echo "$(CYAN)📦 Building profiling runner...$(RESET)"
	@cargo build --profile $(PROFILE_PROFILE) -p $(BENCH_CRATE) --example profile_runner

profile: profile-build ## Profile one dataset with samply (DATASET=dragon_2000.txt VARIANT=vanilla|peel|quot)
	@if [ -z "$(DATASET)" ]; then \
		echo "$(RED)Usage: make profile DATASET=dragon_2000.txt [VARIANT=peel]$(RESET)"; \
		exit 1; \
	fi
	@file="$(DATASET)"; \
	name=$$(basename "$$file" .txt); \
	OUT_DIR="$(PROFILE_DIR)/$(VARIANT)_h$(MAX_DIM)"; \
	mkdir -p "$$OUT_DIR"; \
	out="$$OUT_DIR/$$name.json.gz"; \
	echo "$(CYAN)📊 Profiling $(VARIANT)_h$(MAX_DIM)/$$name...$(RESET)"; \
	$(SAMPLY) record \
		--save-only \
		--unstable-presymbolicate \
		--rate $(PROFILE_RATE) \
		-o "$$out" \
		-- $(PROFILE_TARGET_DIR)/examples/profile_runner "$(MAX_DIM)" "$$file" "$(PROFILE_REPEATS)" "$(VARIANT)"; \
	echo "$(GREEN)✅ Profile written to $$out.$(RESET)"

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
