/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef INCLUDE_CONTEXT_H_
#define INCLUDE_CONTEXT_H_

#include "include/cnnl_core.h"
#include "cn_api.h"

#define CONTEXT_DEVICENAME_BUFFER_SIZE 64
#define CONTEXT_DEVICENAME_LEAST_SIZE 6

/*
Tested version dependency:

| CNNL version | CNTOOLKIT version | CNRT version | DRIVER version  |
---------------------------------------------------------------------
| CNNL V1.2    | CNTOOLKIT V1.6    | CNRT V4.9    | DRIVER V4.8     |
| CNNL V1.1    | CNTOOLKIT V1.5    | CNRT V4.8    | DRIVER V4.7     |
| CNNL V1.0    | CNTOOLKIT V1.4    | CNRT V4.7    | DRIVER V4.6     |
*/
#define CNNL_DEP_CNRT_MIN_MAJOR 5
#define CNNL_DEP_CNRT_MIN_MINOR 0
#define CNNL_DEP_CNRT_MIN_PATCHLEVEL 0

// Compatible with higher version CNRT by default.
#define CNNL_DEP_CNRT_MAX_MAJOR 999
#define CNNL_DEP_CNRT_MAX_MINOR 999
#define CNNL_DEP_CNRT_MAX_PATCHLEVEL 999

typedef enum {
  CNNL_UNKNOWN_DEVICE = 0,
  // CNNL_MLU100 = 100,
  CNNL_MLU220 = 220,
  CNNL_MLU270 = 270,
  CNNL_MLU290 = 290,
} cnnlDevType_t;

struct deviceName {
  char name[CONTEXT_DEVICENAME_BUFFER_SIZE];
  cnnlDevType_t type;
};

struct cnnlContext {
  CNdev device;
  cnrtQueue_t queue;
  cnnlDevType_t arch;  // return arch type. e.g. CNNL_MLU270
  int32_t cluster_num;
  int32_t core_num_per_cluster;
  int32_t nram_size;
  int32_t wram_size;
  int32_t sram_size;
  int32_t capability_cluster_num;
  int32_t capability_job_limit;
};

typedef enum {
  WARNING = 1,
  ERROR   = 2,
} DepCheckLevel;  // related to include/cnlog.h

cnnlStatus_t cnnlCheckDependency(bool need_check_min = true,
                                 bool need_check_max = false,
                                 DepCheckLevel level = WARNING);

#endif  // INCLUDE_CONTEXT_H_
