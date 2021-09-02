# Target rules
all: build

build: libcnnl_example.so

CNSRCS := $(wildcard ./kernels/**/*.mlu)
CNOBJS := $(patsubst %mlu,%o,$(CNSRCS))

export NEUWARE_HOME ?= /usr/local/neuware
INCLUDES := -I$(NEUWARE_HOME)/include/ -I$(CURDIR)
LIBRARIES := -L$(NEUWARE_HOME)/lib64/ -L$(CURDIR)/lib/
CNCCFLAGS := -Wall -fPIC -std=c++11 -pthread --target=x86_64-linux-gnu -O3 --bang-mlu-arch=mtp_220 --bang-mlu-arch=mtp_270 --bang-mlu-arch=mtp_290 -DCNCC
LDFLAGS := -lcnrt -lcndrv -lcnnl_core

libcnnl_example.so: $(CNOBJS)
	$(CXX) -shared -o $@ $+ $(LIBRARIES) $(LDFLAGS)

%.o: %.mlu
	$(NEUWARE_HOME)/bin/cncc $(INCLUDES) $(CNCCFLAGS) -o $@ -c $^

clean:
	rm -rf $(CNOBJS)
	rm -rf libcnnl_example.so

clobber: clean
