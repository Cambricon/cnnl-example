# Target rules
all: build

build: test_example

export NEUWARE_HOME ?= /usr/local/neuware
CNNL_EXAMPLE_DIR=$(CURDIR)/..
OBJS = test_example.o tool.o log.o
INCLUDES := -I$(NEUWARE_HOME)/include -I$(CNNL_EXAMPLE_DIR)/include -I$(CNNL_EXAMPLE_DIR)
LIBRARIES := -L$(NEUWARE_HOME)/lib64 -L$(CNNL_EXAMPLE_DIR)/lib -L$(CNNL_EXAMPLE_DIR)
CXXFLAGS := -Og -std=c++11 -fPIC -lstdc++ -Wall
LDFLAGS := -lcnnl_example -lcnnl_core -lcnrt -lcndrv

test_example: $(OBJS)
	$(CXX) -o $@ $+ $(LIBRARIES) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

clean:
	rm -rf $(OBJS)
	rm -rf test_example

clobber: clean
