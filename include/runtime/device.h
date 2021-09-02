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
#ifndef INCLUDE_RUNTIME_DEVICE_H_
#define INCLUDE_RUNTIME_DEVICE_H_

#include <pthread.h>
#include <string>
#include "include/cnnl_core.h"
#include "cn_api.h"
#include "include/context.h"
#include "include/tensor.h"
#include "include/type.h"

typedef void *MLUaddr;
typedef void *HOSTaddr;

namespace cnnl {
namespace runtime {

#define DEVICE_NAME_LENGTH (64)
inline int32_t getNumOfUnionCapability(cnnlHandle_t handle) {
  return handle->cluster_num;
}
inline int32_t getCoreNumOfEachUnionCapability(cnnlHandle_t handle) {
  return handle->core_num_per_cluster;
}
inline int32_t getNramSizeInBytes(cnnlHandle_t handle) {
  return handle->nram_size;
}
inline int32_t getWramSizeInBytes(cnnlHandle_t handle) {
  return handle->wram_size;
}
inline int32_t getSramSizeInBytes(cnnlHandle_t handle) {
  return handle->sram_size;
}
inline int32_t getClusterLimitCapability(cnnlHandle_t handle) {
  return handle->capability_cluster_num;
}
inline int32_t getJobLimitCapability(cnnlHandle_t handle) {
  return handle->capability_job_limit;
}
}  // namespace runtime
}  // namespace cnnl

#endif  // INCLUDE_RUNTIME_DEVICE_H_
