DEBUG ?= 0
OPTIMIZE ?= 1

CC = gcc
CFLAGS = -Wall -Wextra -fopenmp
LDFLAGS = -lmatio -lm -fopenmp

ifeq ($(DEBUG),1)
CFLAGS += -g3 -DDEBUG
else ifeq ($(OPTIMIZE),1)
CFLAGS += -O3 -march=native
else
CFLAGS += -O2
endif

BUILD_DIR = build
BIN_DIR = bin

CSR_SOURCES = spMM/csr_matmat.c spMV/csr_spmv.c
PCSR_SOURCES = spMM/pcsr_matmat.c spMV/pcsr_spmv.c
BCSR_SOURCES = spMM/bcsr_matmat.c spMV/bcsr_spmv.c
COO_SOURCES = spMM/coo_matmat.c spMV/coo_spmv.c
ELL_SOURCES = spMM/ell_matmat.c spMV/ell_spmv.c
HYBRID_SOURCES = spMM/hybrid_matmat.c spMV/hybrid_spmv.c

CSR_OBJECTS = $(CSR_SOURCES:%.c=$(BUILD_DIR)/%.o)
PCSR_OBJECTS = $(PCSR_SOURCES:%.c=$(BUILD_DIR)/%.o)
BCSR_OBJECTS = $(BCSR_SOURCES:%.c=$(BUILD_DIR)/%.o)
COO_OBJECTS = $(COO_SOURCES:%.c=$(BUILD_DIR)/%.o)
ELL_OBJECTS = $(ELL_SOURCES:%.c=$(BUILD_DIR)/%.o)
HYBRID_OBJECTS = $(HYBRID_SOURCES:%.c=$(BUILD_DIR)/%.o)

CSR_EXECS = $(CSR_SOURCES:%.c=$(BIN_DIR)/%)
PCSR_EXECS = $(PCSR_SOURCES:%.c=$(BIN_DIR)/%)
BCSR_EXECS = $(BCSR_SOURCES:%.c=$(BIN_DIR)/%)
COO_EXECS = $(COO_SOURCES:%.c=$(BIN_DIR)/%)
ELL_EXECS = $(ELL_SOURCES:%.c=$(BIN_DIR)/%)
HYBRID_EXECS = $(HYBRID_SOURCES:%.c=$(BIN_DIR)/%)

ALL_EXECS = $(CSR_EXECS) $(PCSR_EXECS) $(BCSR_EXECS) $(COO_EXECS) $(ELL_EXECS) $(HYBRID_EXECS)

$(shell mkdir -p $(BUILD_DIR)/spMM $(BUILD_DIR)/spMV $(BIN_DIR))

all: $(ALL_EXECS)

csr: $(CSR_EXECS)
pcsr: $(PCSR_EXECS)
bcsr: $(BCSR_EXECS)
coo: $(COO_EXECS)
ell: $(ELL_EXECS)
hybrid: $(HYBRID_EXECS)

spmv: $(BIN_DIR)/csr_spmv $(BIN_DIR)/pcsr_spmv $(BIN_DIR)/bcsr_spmv $(BIN_DIR)/coo_spmv $(BIN_DIR)/ell_spmv $(BIN_DIR)/hybrid_spmv

spmm: $(BIN_DIR)/csr_matmat $(BIN_DIR)/pcsr_matmat $(BIN_DIR)/bcsr_matmat $(BIN_DIR)/coo_matmat $(BIN_DIR)/ell_matmat $(BIN_DIR)/hybrid_matmat

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/%: $(BUILD_DIR)/%.o
	$(CC) $< -o $@ $(LDFLAGS)

benchmark:
	@for exec in $(wildcard $(BIN_DIR)/*); do \
	echo "\nRunning $$exec..."; \
	$$exec $(TEST_MATRIX) $(TEST_MATRIX2); \
	done

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

config:
	@echo "Build Configuration:"
	@echo "  Debug: $(DEBUG)"
	@echo "  Optimize: $(OPTIMIZE)"
	@echo "  CFLAGS: $(CFLAGS)"

help:
	@echo "Available targets:"
	@echo "  all    - Build all formats"
	@echo "  csr    - Build CSR format"
	@echo "  pcsr   - Build PCSR format"
	@echo "  bcsr   - Build BCSR format"
	@echo "  coo    - Build COO format"
	@echo "  ell    - Build ELL format"
	@echo "  hybrid - Build Hybrid format"
	@echo "  spmv   - Build all SpMV programs"
	@echo "  spmm   - Build all SpMM programs"
	@echo ""
	@echo "Build options:"
	@echo "  DEBUG=1     - Enable debug build"
	@echo "  OPTIMIZE=0  - Disable optimizations"

.PHONY: all clean benchmark config help csr pcsr bcsr coo ell hybrid spmv spmm
