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
#ifndef INCLUDE_GET_HARDWARE_TIME_H_
#define INCLUDE_GET_HARDWARE_TIME_H_

#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <vector>
#include "include/cnnl_core.h"
#include "include/tensor.h"
#include "include/type.h"
#include "include/logging.h"

#define GET_HARDWARE_TIME_START(queue, func) \
  cnrtNotifier_t notifier_start = cnnl::get_hardware_time::getHardwareTimeStart(queue, func)
#define GET_HARDWARE_TIME_END(queue, func) \
  cnnl::get_hardware_time::getHardwareTimeEnd(queue, func, notifier_start)

namespace cnnl {
namespace get_hardware_time {

bool getBoolEnvVar(const std::string &str, bool default_para = false);
std::string getStringEnvVar(const std::string &str, std::string default_para);
bool getBoolOpName(const std::string str, const std::string op_name);
cnrtNotifier_t getHardwareTimeStart(cnrtQueue_t queue, std::string func);
void getHardwareTimeEnd(cnrtQueue_t queue, std::string func, cnrtNotifier_t notifier_start);
}  // namespace get_hardware_time
}  // namespace cnnl
#endif  // INCLUDE_GET_HARDWARE_TIME_H_
