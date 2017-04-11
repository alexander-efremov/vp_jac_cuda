NVCC := /usr/local/cuda-7.0/bin/nvcc -ccbin g++
BIN := .

# internal flags
ifeq ($(dbg), 1)
	NVCCFLAGS   := -g -G -DDEBUG 
else
	NVCCFLAGS += -m64 -O2 
endif

CCFLAGS     := -O2
LDFLAGS     := 

# Extra user flags
EXTRA_NVCCFLAGS   ?= --ptxas-options=-v
EXTRA_LDFLAGS     ?= 
EXTRA_CCFLAGS     ?=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES  := -I./
LIBRARIES := 

################################################################################

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_FLAGS   ?= $(GENCODE_SM20)

################################################################################

OBJ := obj/cuda

# Target rules
all: build

build: jacobi

$(OBJ)/jacobi.o: jacobi.cu
	-mkdir -p $(OBJ)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ)/timer.o: timer.cpp
	-mkdir -p $(OBJ)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


jacobi: $(OBJ)/timer.o $(OBJ)/jacobi.o 
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f jacobi $(OBJ)/*.o

clobber: clean
