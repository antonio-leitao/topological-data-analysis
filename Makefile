.PHONY: help install build-py build-rust test test-core test-dense test-sparse bench bench-dense bench-sparse bench-save profile profile-build profile-dense profile-sparse profile-dense-all profile-sparse-all publish-py publish-rust clean
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
PROFILE_RATE ?= 1000
PROFILE_DIR ?= target/profiles
PROFILE_H2_LAST_DATASET ?= hiv1.txt
PROFILE_REPEATS ?= 1
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

test-dense: install ## Run Python dense correctness test
	@echo "$(CYAN)🧪 Running Python dense correctness...$(RESET)"
	@$(PYTHON) crates/python/tests/correctness.py dense $(MAX_DIM)

test-sparse: install ## Run Python sparse correctness test
	@echo "$(CYAN)🧪 Running Python sparse correctness...$(RESET)"
	@$(PYTHON) crates/python/tests/correctness.py sparse $(MAX_DIM)

test: test-core test-dense test-sparse ## Run all correctness tests
	@echo "$(GREEN)✅ Tests passed!$(RESET)"

# ── Bench ─────────────────────────────────────────────────────────────────

bench-dense: ## Run dense Rust core benchmarks
	@echo "$(CYAN)📊 Running dense Rust core benchmarks...$(RESET)"
	@TDA_MAX_DIM=$(MAX_DIM) TDA_BENCH_MODE=dense cargo bench -p $(BENCH_CRATE) --bench $(BENCH_NAME)

bench-sparse: ## Run sparse Rust core benchmarks
	@echo "$(CYAN)📊 Running sparse Rust core benchmarks...$(RESET)"
	@TDA_MAX_DIM=$(MAX_DIM) TDA_BENCH_MODE=sparse cargo bench -p $(BENCH_CRATE) --bench $(BENCH_NAME)

bench: ## Run dense and sparse Rust core benchmarks
	@echo "$(CYAN)📊 Running dense and sparse Rust core benchmarks...$(RESET)"
	@TDA_MAX_DIM=$(MAX_DIM) TDA_BENCH_MODE=both cargo bench -p $(BENCH_CRATE) --bench $(BENCH_NAME)

profile-build: ## Build the single-run profiling runner
	@echo "$(CYAN)📦 Building profiling runner...$(RESET)"
	@cargo build --release -p $(BENCH_CRATE) --example profile_runner

profile-dense: profile-build ## Profile one dense dataset with samply (DATASET=dragon_2000.txt)
	@if [ -z "$(DATASET)" ]; then \
		echo "$(RED)Usage: make profile-dense DATASET=dragon_2000.txt$(RESET)"; \
		exit 1; \
	fi
	@file="$(DATASET)"; \
	name=$$(basename "$$file" .txt); \
	OUT_DIR="$(PROFILE_DIR)/dense_h$(MAX_DIM)"; \
	mkdir -p "$$OUT_DIR"; \
	out="$$OUT_DIR/$$name.json.gz"; \
	echo "$(CYAN)📊 Profiling dense_h$(MAX_DIM)/$$name...$(RESET)"; \
	$(SAMPLY) record \
		--save-only \
		--unstable-presymbolicate \
		--main-thread-only \
		--rate $(PROFILE_RATE) \
		-o "$$out" \
		-- target/release/examples/profile_runner dense "$(MAX_DIM)" "$$file" "$(PROFILE_REPEATS)"; \
	echo "$(GREEN)✅ Dense profile written to $$out.$(RESET)"

profile-sparse: profile-build ## Profile one sparse dataset with samply (DATASET=dragon_2000.txt)
	@if [ -z "$(DATASET)" ]; then \
		echo "$(RED)Usage: make profile-sparse DATASET=dragon_2000.txt$(RESET)"; \
		exit 1; \
	fi
	@file="$(DATASET)"; \
	name=$$(basename "$$file" .txt); \
	OUT_DIR="$(PROFILE_DIR)/sparse_h$(MAX_DIM)"; \
	mkdir -p "$$OUT_DIR"; \
	out="$$OUT_DIR/$$name.json.gz"; \
	echo "$(CYAN)📊 Profiling sparse_h$(MAX_DIM)/$$name...$(RESET)"; \
	$(SAMPLY) record \
		--save-only \
		--unstable-presymbolicate \
		--main-thread-only \
		--rate $(PROFILE_RATE) \
		-o "$$out" \
		-- target/release/examples/profile_runner sparse "$(MAX_DIM)" "$$file" "$(PROFILE_REPEATS)"; \
	echo "$(GREEN)✅ Sparse profile written to $$out.$(RESET)"

profile-dense-all: profile-build ## Profile every dense dataset with samply
	@OUT_DIR="$(PROFILE_DIR)/dense_h$(MAX_DIM)"; \
	mkdir -p "$$OUT_DIR"; \
	DATASETS=$$(awk -v max_dim="$(MAX_DIM)" -v cutoff="$(PROFILE_H2_LAST_DATASET)" 'NF && $$1 !~ /^#/ { print $$1; if (max_dim == 2 && $$1 == cutoff) exit }' data/datasets.txt); \
	for file in $$DATASETS; do \
		name=$${file%.txt}; \
		out="$$OUT_DIR/$$name.json.gz"; \
		echo "$(CYAN)📊 Profiling dense_h$(MAX_DIM)/$$name...$(RESET)"; \
		$(SAMPLY) record \
			--save-only \
			--unstable-presymbolicate \
			--main-thread-only \
			--rate $(PROFILE_RATE) \
			-o "$$out" \
			-- target/release/examples/profile_runner dense "$(MAX_DIM)" "$$file" "$(PROFILE_REPEATS)"; \
	done; \
	echo "$(GREEN)✅ Dense profiles written to $$OUT_DIR.$(RESET)"

profile-sparse-all: profile-build ## Profile every sparse dataset with samply
	@OUT_DIR="$(PROFILE_DIR)/sparse_h$(MAX_DIM)"; \
	mkdir -p "$$OUT_DIR"; \
	DATASETS=$$(awk -v max_dim="$(MAX_DIM)" -v cutoff="$(PROFILE_H2_LAST_DATASET)" 'NF && $$1 !~ /^#/ { print $$1; if (max_dim == 2 && $$1 == cutoff) exit }' data/datasets.txt); \
	for file in $$DATASETS; do \
		name=$${file%.txt}; \
		out="$$OUT_DIR/$$name.json.gz"; \
		echo "$(CYAN)📊 Profiling sparse_h$(MAX_DIM)/$$name...$(RESET)"; \
		$(SAMPLY) record \
			--save-only \
			--unstable-presymbolicate \
			--main-thread-only \
			--rate $(PROFILE_RATE) \
			-o "$$out" \
			-- target/release/examples/profile_runner sparse "$(MAX_DIM)" "$$file" "$(PROFILE_REPEATS)"; \
	done; \
	echo "$(GREEN)✅ Sparse profiles written to $$OUT_DIR.$(RESET)"

profile: profile-dense-all profile-sparse-all ## Profile dense and sparse Rust core benchmarks with samply
	@echo "$(GREEN)✅ Profiles complete.$(RESET)"

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
