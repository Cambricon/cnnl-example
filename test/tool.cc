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
#include <vector>
#include <random>
#include <limits>
#include "cnnl_example.h"
#include "string.h"
#include "log.h"
#include "tool.h"

bool isBeginWith(const char *str1, const char *str2) {
  if (str1 == NULL || str2 == NULL) {
    return false;
  }
  int len1 = strlen(str1);
  int len2 = strlen(str2);
  if ((len1 < len2) || (len1 == 0 || len2 == 0)) {
    return false;
  }
  const char *p = str2;
  int i = 0;
  while (*p != '\0') {
    if (*p != str1[i]) {
      return false;
    } else {
      p++;
      i++;
    }
  }
  return true;
}

float *mallocDataRandf(int size, int low, int height) {
  float *data = (float *)malloc(size * sizeof(float));
  std::uniform_real_distribution<float> dist(low, height);
  std::default_random_engine random(time(NULL));
  for (int i = 0; i < size; i++) {
    data[i] = dist(random);
  }
  return data;
}

std::string name2Str(const OpName &op_name) {
  switch (op_name) {
    case CNNL_ABS:
      return "cnnlAbs";
    case CNNL_LOG:
      return "cnnlLog";
    case CNNL_DIV:
      return "cnnlDiv";
    case CNNL_SQRT:
      return "cnnlSqrt";
    case CNNL_SQRT_BACKWARD:
      return "cnnlSqrtBackward";
    default:
      return "unkonw";
  }
}

// get --op_name argument's value
void getOpName(const char *name_str, ParamInfo &param_info) {
  if (strcmp(name_str, "cnnlAbs") == 0) {
    param_info.op_name = CNNL_ABS;
    param_info.input_num = 1;
  } else if (strcmp(name_str, "cnnlLog") == 0) {
    param_info.op_name = CNNL_LOG;
    param_info.input_num = 1;
  } else if (strcmp(name_str, "cnnlDiv") == 0) {
    param_info.op_name = CNNL_DIV;
    param_info.input_num = 2;
  } else if (strcmp(name_str, "cnnlSqrt") == 0) {
    param_info.op_name = CNNL_SQRT;
    param_info.input_num = 1;
  } else if (strcmp(name_str, "cnnlSqrtBackward") == 0) {
    param_info.op_name = CNNL_SQRT_BACKWARD;
    param_info.input_num = 2;
  } else {
    std::string name = name_str;
    std::string msg = "unsupprt name:" + name;
    ERROR(msg);
  }
}

// get --input_shape and --output_shape argument's value
void getShape(const char *argv, const bool is_input, ParamInfo &param_info) {
  char *p1 = (char *)malloc(strlen(argv));
  p1 = strncpy(p1, argv + 1, strlen(argv) - 2);  // remove "{" and "}"
  char *begin = p1;
  char number[20];
  int index = 0;
  while ((p1 = (char *)strchr(begin, '-')) != NULL) {
    int data_len = strlen(begin) - strlen(p1);
    strncpy(number, begin, strlen(begin) - strlen(p1));
    number[data_len] = '\0';
    begin = p1 + 1;
    if (is_input) {
      param_info.input_shape[index] = atol(number);
    } else {
      param_info.output_shape[index] = atol(number);
    }
    index++;
    if (index > 7) {
      ERROR("dim size of shape unsupported, max dim size supported is 8!")
      return;
    }
  }
  if (is_input) {
    param_info.input_shape[index] = atol(begin);
  } else {
    param_info.output_shape[index] = atol(begin);
  }
  param_info.dim_size = index + 1;
}

// get --data_type argument's value
void getDataType(const char *data_type, cnnlDataType_t &dtype) {
  if (strcmp(data_type, "half") == 0) {
    dtype = CNNL_DTYPE_HALF;
  } else if (strcmp(data_type, "float") == 0) {
    dtype = CNNL_DTYPE_FLOAT;
  } else {
    std::string type = data_type;
    std::string msg = "unsupported data type:" + type;
    ERROR(msg);
  }
}

// get --prefer argument's value
void getPreferValue(const char *prefer_value, cnnlComputationPreference_t &prefer) {
  if (strcmp(prefer_value, "fast") == 0) {
    prefer = CNNL_COMPUTATION_FAST;
  } else if (strcmp(prefer_value, "accuracy") == 0) {
    prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  } else {
    std::string param = prefer_value;
    std::string msg = "unsupported prefer mode:" + param;
    ERROR(msg);
  }
}

// get --log_base argument's value
void getLogBaseValue(const char *log_base, cnnlLogBase_t &base_value) {
  if (strcmp(log_base, "2") == 0) {
    base_value = CNNL_LOG_2;
  } else if (strcmp(log_base, "10") == 0) {
    base_value = CNNL_LOG_10;
  } else if (strcmp(log_base, "e") == 0) {
    base_value = CNNL_LOG_E;
  } else {
    std::string param = log_base;
    std::string msg = "unsupported log base:" + param;
    ERROR(msg);
  }
}

// parse command line arguments
void parseParam(int argc, char *argv[], ParamInfo &param_info) {
  if (argc != 5 && argc != 6 && argc != 7) {
    std::stringstream error_msg;
    error_msg
        << "wrong command line arguments, please reference to the example in run_test_example.sh!"
        << std::endl;
    ERROR(error_msg.str());
  }
  argc -= 1;
  argv++;
  while (argc) {
    if (isBeginWith(argv[0], "--op_name")) {
      getOpName(argv[0] + 10, param_info);
    } else if (isBeginWith(argv[0], "--input_shape")) {
      getShape(argv[0] + 14, true, param_info);
    } else if (isBeginWith(argv[0], "--output_shape")) {
      getShape(argv[0] + 15, false, param_info);
    } else if (isBeginWith(argv[0], "--data_type")) {
      getDataType(argv[0] + 12, param_info.dtype);
    } else if (isBeginWith(argv[0], "--prefer")) {
      getPreferValue(argv[0] + 9, param_info.prefer);
    } else if (isBeginWith(argv[0], "--log_base")) {
      getLogBaseValue(argv[0] + 11, param_info.log_base);
    } else {
      std::string opt_param = argv[0];
      std::string error_message = "unsupported param:" + opt_param;
      ERROR(error_message);
    }
    argc -= 1;
    argv++;
  }
}

// show case info
void printTestParam(const ParamInfo &param_info) {
  std::stringstream case_info;
  std::string dtype = param_info.dtype == 1 ? "CNNL_DTYPE_HALF" : "CNNL_DTYPE_FLOAT";
  case_info << "-----------------test case info-----------------" << std::endl;
  case_info << "op name:" << name2Str(param_info.op_name) << std::endl;
  case_info << "data type:" << dtype << std::endl;
  case_info << "tensor shape:" << param_info.input_shape[0];
  for (int i = 1; i < param_info.dim_size; ++i) {
    case_info << ", " << param_info.input_shape[i];
  }
  case_info << std::endl;
  std::cout << case_info.str();
}

// init device resources
void initDevice(cnrtDev_t &dev, cnrtQueue_t &queue, cnnlHandle_t &handle) {
  CNRT_CHECK(cnrtInit(0));
  // cnrt: get current device 0
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  // cnrt: create queue
  CNRT_CHECK(cnrtCreateQueue(&queue));

  // cnnl: create handle and bind queue
  CNNL_CHECK(cnnlCreate(&handle));
  CNNL_CHECK(cnnlSetQueue(handle, queue));
}

// create input and  output tensor descriptor for compute interface
void createDeviceDesc(const ParamInfo &param_info, BaseOp &base_op) {
  size_t element_num = 1;
  for (int i = 0; i < param_info.dim_size; ++i) {
    element_num *= param_info.input_shape[i];
  }
  for (int i = 0; i < param_info.input_num; i++) {
    cnnlTensorDescriptor_t temp_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&temp_desc));
    CNNL_CHECK(cnnlSetTensorDescriptor(temp_desc, CNNL_LAYOUT_ARRAY, param_info.dtype,
                                       param_info.dim_size, param_info.input_shape));
    base_op.inputs.push_back(temp_desc);
  }
  for (int i = 0; i < param_info.output_num; i++) {
    cnnlTensorDescriptor_t temp_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&temp_desc));
    CNNL_CHECK(cnnlSetTensorDescriptor(temp_desc, CNNL_LAYOUT_ARRAY, param_info.dtype,
                                       param_info.dim_size, param_info.output_shape));
    base_op.outputs.push_back(temp_desc);
  }
}

size_t getDataTypeSize(const cnnlDataType_t dtype) {
  switch (dtype) {
    case CNNL_DTYPE_HALF:
      return 2;
    case CNNL_DTYPE_FLOAT:
      return 4;
    default:
      return -1;
  }
}

// perpare test data and device memory, then copy data to device
void prepareTestData(const ParamInfo &param_info, BaseOp &base_op) {
  int low = -1, height = 1;
  size_t element_num = 1;
  for (int i = 0; i < param_info.dim_size; ++i) {
    element_num *= param_info.input_shape[i];
  }
  int tensor_size = element_num * getDataTypeSize(param_info.dtype);

  for (int i = 0; i < param_info.input_num + param_info.output_num; i++) {
    DataAddrInfo data_node;
    data_node.size = tensor_size;
    data_node.host_ptr = mallocDataRandf(element_num, low, height);
    CNRT_CHECK(cnrtMalloc(&(data_node.device_ptr), tensor_size));
    CNRT_CHECK(cnrtMemset(data_node.device_ptr, 0, tensor_size));
    base_op.datas.push_back(data_node);
  }

  // copy input from host to device
  for (int i = 0; i < param_info.input_num; ++i) {
    if (param_info.dtype == CNNL_DTYPE_HALF) {
      // convert float32 data to half type for device compute
      char *temp_half = (char *)malloc(element_num * 2 * sizeof(char));
      CNRT_CHECK(cnrtCastDataType(base_op.datas[i].host_ptr, CNRT_FLOAT32, temp_half, CNRT_FLOAT16,
                                  element_num, NULL));
      CNRT_CHECK(cnrtMemcpy(base_op.datas[i].device_ptr, temp_half, tensor_size,
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
      free(temp_half);
    } else {
      CNRT_CHECK(cnrtMemcpy(base_op.datas[i].device_ptr, base_op.datas[i].host_ptr, tensor_size,
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    }
  }
}

// call operation compute interface and sync task queue
void deviceCompute(const cnnlHandle_t &handle,
                   const cnrtQueue_t &queue,
                   const ParamInfo &param_info,
                   const BaseOp &base_op) {
  switch (param_info.op_name) {
    case CNNL_ABS:
      CNNL_CHECK(cnnlAbs(handle, base_op.inputs[0], base_op.datas[0].device_ptr, base_op.outputs[0],
                         base_op.datas[1].device_ptr));
      CNRT_CHECK(cnrtSyncQueue(queue));
      break;
    case CNNL_LOG:
      CNNL_CHECK(cnnlLog(handle, param_info.prefer, param_info.log_base, base_op.inputs[0],
                         base_op.datas[0].device_ptr, base_op.outputs[0],
                         base_op.datas[1].device_ptr));
      CNRT_CHECK(cnrtSyncQueue(queue));
      break;
    case CNNL_SQRT:
      CNNL_CHECK(cnnlSqrt(handle, param_info.prefer, base_op.inputs[0], base_op.datas[0].device_ptr,
                          base_op.outputs[0], base_op.datas[1].device_ptr));
      CNRT_CHECK(cnrtSyncQueue(queue));
      break;
    case CNNL_DIV:
      CNNL_CHECK(cnnlDiv(handle, param_info.prefer, base_op.inputs[0], base_op.datas[0].device_ptr,
                         base_op.inputs[1], base_op.datas[1].device_ptr, base_op.outputs[0],
                         base_op.datas[2].device_ptr));
      CNRT_CHECK(cnrtSyncQueue(queue));
      break;
    case CNNL_SQRT_BACKWARD:
      CNNL_CHECK(cnnlSqrtBackward(handle, base_op.inputs[0], base_op.datas[0].device_ptr,
                                  base_op.inputs[1], base_op.datas[1].device_ptr,
                                  base_op.outputs[0], base_op.datas[2].device_ptr));
      CNRT_CHECK(cnrtSyncQueue(queue));
      break;
    default:
      return;
  }
}

// copy result from device memory to host
void copyResultOut(const ParamInfo &param_info, BaseOp &base_op) {
  DataAddrInfo output_node = base_op.datas[param_info.input_num];
  if (param_info.dtype == CNNL_DTYPE_HALF) {
    // convert device half result to float32
    char *temp_half = (char *)malloc(output_node.size);
    CNRT_CHECK(cnrtMemcpy(temp_half, output_node.device_ptr, output_node.size,
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtCastDataType(temp_half, CNRT_FLOAT16, output_node.host_ptr, CNRT_FLOAT32,
                                output_node.size / 2, NULL));
  } else {
    CNRT_CHECK(cnrtMemcpy(output_node.host_ptr, output_node.device_ptr, output_node.size,
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
  }
}

// free resources after compute
void deviceAndHostFree(cnnlHandle_t &handle, cnrtQueue_t &queue, BaseOp &base_op) {
  // destroy tensor descriptor
  for (auto &tensor : base_op.inputs) {
    CNNL_CHECK(cnnlDestroyTensorDescriptor(tensor));
  }
  for (auto &tensor : base_op.outputs) {
    CNNL_CHECK(cnnlDestroyTensorDescriptor(tensor));
  }

  // free device and host memory
  for (auto &data : base_op.datas) {
    CNRT_CHECK(cnrtFree(data.device_ptr));
    free(data.host_ptr);
  }

  // destroy queue and runtime context
  CNRT_CHECK(cnrtDestroyQueue(queue));
  CNNL_CHECK(cnnlDestroy(handle));
}
