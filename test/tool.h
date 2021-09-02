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
#ifndef TEST_TOOL_H_
#define TEST_TOOL_H_

#include <vector>
#include "cnnl_example.h"
#define PARAM_NUM 22
#define MAX_DIM 8
#define ARGC_NUM 5

enum OpName { CNNL_ABS = 0, CNNL_LOG = 1, CNNL_SQRT = 2, CNNL_SQRT_BACKWARD = 3, CNNL_DIV = 4 };

struct ParamInfo {
  int input_shape[MAX_DIM];
  int output_shape[MAX_DIM];
  int input_num  = 1;
  int output_num = 1;
  int dim_size;
  cnnlDataType_t dtype;
  OpName op_name;
  cnnlLogBase_t log_base;
  cnnlComputationPreference_t prefer;
};

struct DataAddrInfo {
  float *host_ptr  = NULL;
  void *device_ptr = NULL;
  size_t size;
};

struct BaseOp {
  std::vector<cnnlTensorDescriptor_t> inputs;
  std::vector<cnnlTensorDescriptor_t> outputs;
  std::vector<DataAddrInfo> datas;
};

void parseParam(int argc, char *argv[], ParamInfo &param_info);

void printTestParam(const ParamInfo &param_info);

void initDevice(cnrtDev_t &dev, cnrtQueue_t &queue, cnnlHandle_t &handle);

void createDeviceDesc(const ParamInfo &param_info, BaseOp &base_op);

void prepareTestData(const ParamInfo &param_info, BaseOp &base_op);

void deviceCompute(const cnnlHandle_t &handle,
                   const cnrtQueue_t &queue,
                   const ParamInfo &param_info,
                   const BaseOp &base_op);

void copyResultOut(const ParamInfo &param_info, BaseOp &base_op);

void deviceAndHostFree(cnnlHandle_t &handle, cnrtQueue_t &queue, BaseOp &base_op);
#endif  // TEST_TOOL_H_
