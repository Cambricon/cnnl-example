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
#ifndef INCLUDE_TENSOR_H_
#define INCLUDE_TENSOR_H_

#include <vector>
#include <list>
#include <memory>
#include <queue>
#include <thread>  // NOLINT
#include <atomic>
#include <cstring>
#include "include/cnnl_core.h"
#include "include/macros.h"
#include "include/logging.h"
#include "include/type.h"

#define QUEUE_ARRAY_LENGTH 4

struct cnnlTensorStruct {
  cnnlTensorStruct()
      : dim(0),
        dtype(CNNL_DTYPE_FLOAT),
        onchip_dtype(CNNL_DTYPE_INVALID),
        layout(CNNL_LAYOUT_ARRAY),
        position(0),
        scale(1.0),
        offset(0) {
    /* explicit set initial values for document use.
     */
  }
  ~cnnlTensorStruct() {
    /* please do NOT implement any codes here.
     * a state-less struct should not hold any resources.
     */
  }
  /* methods */
  cnnlStatus_t tensorDimN(size_t &dim);
  cnnlStatus_t tensorDimC(size_t &dim);
  cnnlStatus_t tensorDimH(size_t &dim);
  cnnlStatus_t tensorDimW(size_t &dim);
  inline cnnlStatus_t tensorElementsNumber(size_t &elements) const {
    elements = total_element_num;
    return CNNL_STATUS_SUCCESS;
  }
  inline cnnlStatus_t tensorSize(size_t &tensor_size) const {
    tensor_size = total_tensor_size;
    return CNNL_STATUS_SUCCESS;
  }

  /* struct */
  int dim               = 0;
  int total_element_num = 0;
  int total_tensor_size = 0;
  // if dimNb > CNNL_DIM_MAX (8), using larger_dims, malloc it and dims point it.
  // else, using normal_dims, dont need malloc and free.
  int normal_dims[CNNL_DIM_MAX] = {-1};
  int *larger_dims              = NULL;
  int *dims                     = normal_dims;  // point the normal dims as default

  int normal_strides[CNNL_DIM_MAX] = {-1};
  int *larger_strides              = NULL;
  int *strides                     = normal_strides;  // point the normal strides as default

  cnnlDataType_t dtype;
  cnnlDataType_t onchip_dtype;
  cnnlTensorLayout_t layout;
  int position;
  float scale;
  int offset;
  int channelNb;
  std::vector<int> positions;
  std::vector<float> scales;
  std::vector<int> offsets;
  inline void init() {  // reset random value after malloc.
    // init these pointer.
    // if not, when call reset() will free invalid pointer.
    larger_dims    = NULL;
    larger_strides = NULL;

    dim               = 0;
    total_element_num = 0;
    total_tensor_size = 0;
    dims              = normal_dims;
    strides           = normal_strides;
  }
  inline void reset() {  // reset variable as default.
    if (CNNL_PREDICT_FALSE(larger_dims != NULL)) {
      delete[] larger_dims;
      larger_dims = NULL;
    }
    if (CNNL_PREDICT_FALSE(larger_strides != NULL)) {
      delete[] larger_strides;
      larger_strides = NULL;
    }
    dims         = normal_dims;
    strides      = normal_strides;
    dtype        = CNNL_DTYPE_FLOAT;
    onchip_dtype = CNNL_DTYPE_INVALID;
    layout       = CNNL_LAYOUT_ARRAY;

    position = 0;
    scale    = 1.0f;
    offset   = 0;

    dim               = 0;
    total_element_num = 0;
    total_tensor_size = 0;
  }
};

inline int cnnlDataTypeBytes(const cnnlDataType_t dt) {
  switch (dt) {
    case CNNL_DTYPE_HALF:
      return 2;
    case CNNL_DTYPE_FLOAT:
      return 4;
    case CNNL_DTYPE_INT8:
    case CNNL_DTYPE_UINT8:
    case CNNL_DTYPE_BOOL:
      return 1;
    case CNNL_DTYPE_INT16:
      return 2;
    // case CNNL_DTYPE_INT23:   return 3;
    case CNNL_DTYPE_INT31:
      return 4;
    case CNNL_DTYPE_INT32:
      return 4;
    default:
      return -1;
  }
}

inline int cnnlGetTensordimN(const cnnlTensorDescriptor_t desc) {
  switch (desc->layout) {
    case CNNL_LAYOUT_NCHW:
    case CNNL_LAYOUT_NHWC:
    case CNNL_LAYOUT_NDHWC:
      return desc->dims[0];
    case CNNL_LAYOUT_NCDHW:
      return desc->dims[0];
    case CNNL_LAYOUT_HWCN:
      return desc->dims[3];
    default:
      LOG(ERROR) << "Failed to call dimN, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int cnnlGetTensordimD(const cnnlTensorDescriptor_t desc) {
  switch (desc->layout) {
    case CNNL_LAYOUT_NDHWC:
      return desc->dims[1];
    case CNNL_LAYOUT_NCDHW:
      return desc->dims[2];
    default:
      LOG(ERROR) << "Failed to call dimD, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int cnnlGetTensordimC(const cnnlTensorDescriptor_t desc) {
  switch (desc->layout) {
    case CNNL_LAYOUT_NCHW:
      return desc->dims[1];
    case CNNL_LAYOUT_NHWC:
      return desc->dims[3];
    case CNNL_LAYOUT_NDHWC:
      return desc->dims[4];
    case CNNL_LAYOUT_NCDHW:
      return desc->dims[1];
    case CNNL_LAYOUT_HWCN:
      return desc->dims[2];
    default:
      LOG(ERROR) << "Failed to call dimC, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int cnnlGetTensordimH(const cnnlTensorDescriptor_t desc) {
  switch (desc->layout) {
    case CNNL_LAYOUT_NCHW:
      return desc->dims[2];
    case CNNL_LAYOUT_NHWC:
      return desc->dims[1];
    case CNNL_LAYOUT_NDHWC:
      return desc->dims[2];
    case CNNL_LAYOUT_NCDHW:
      return desc->dims[3];
    case CNNL_LAYOUT_HWCN:
      return desc->dims[0];
    default:
      LOG(ERROR) << "Failed to call dimH, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int cnnlGetTensordimW(const cnnlTensorDescriptor_t desc) {
  switch (desc->layout) {
    case CNNL_LAYOUT_NCHW:
      return desc->dims[3];
    case CNNL_LAYOUT_NHWC:
      return desc->dims[2];
    case CNNL_LAYOUT_NDHWC:
      return desc->dims[3];
    case CNNL_LAYOUT_NCDHW:
      return desc->dims[4];
    case CNNL_LAYOUT_HWCN:
      return desc->dims[1];
    default:
      LOG(ERROR) << "Failed to call dimW, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

#endif  // INCLUDE_TENSOR_H_
