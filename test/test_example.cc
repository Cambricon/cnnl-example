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
#include <stdexcept>
#include <iostream>
#include "tool.h"
#include "log.h"

int main(int argc, char *argv[]) {
  try {
    BaseOp base_op;
    ParamInfo param_info;
    parseParam(argc, argv, param_info);
    printTestParam(param_info);

    // step1: init device
    LOG("init device:dev, handle and queue.");
    cnrtDev_t dev;
    cnrtQueue_t queue = NULL;
    cnnlHandle_t handle = NULL;
    initDevice(dev, queue, handle);

    // step2: prepare device desc include:input/output tensors, op_desc, ...
    LOG("create tensor descriptor and operation descriptor.");
    createDeviceDesc(param_info, base_op);

    // step3: device memory malloc and copy data from host to device
    LOG("malloc device memory and copy input data in.");
    prepareTestData(param_info, base_op);

    // step4: call device compute interface to finish compute
    LOG("begin device compute task.");
    deviceCompute(handle, queue, param_info, base_op);

    // step5: copy result from device to host
    LOG("finish device compute, and copy result out.");
    copyResultOut(param_info, base_op);

    // step6: free device resource
    LOG("free host and device resources.");
    deviceAndHostFree(handle, queue, base_op);

    LOG("example run success!");
  } catch (std::runtime_error &e) {
    ERROR(e.what());
    return -1;
  }
  return 0;
}
