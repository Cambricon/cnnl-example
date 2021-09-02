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
#ifndef KERNELS_BINARY_OP_BINARY_OP_HOST_H_
#define KERNELS_BINARY_OP_BINARY_OP_HOST_H_

#include <string>
#include "include/cnnl_core.h"

void binaryOpPolicyFunc(const cnnlHandle_t &handle,
                        const cnnlTensorDescriptor_t &desc,
                        const int &align_param,
                        cnrtDim3_t *k_dim,
                        cnrtFunctionType_t *k_type);

/* user param check
 * step1:check desc and data ptr is not nullptr_t
 * step2:check shape and data type
 * */
cnnlStatus_t binaryOpParamCheck(const std::string &op_name,
                                const cnnlHandle_t &handle,
                                const cnnlTensorDescriptor_t &input1_desc,
                                const void *input1,
                                const cnnlTensorDescriptor_t &input2_desc,
                                const void *input2,
                                const cnnlTensorDescriptor_t &output_desc,
                                const void *output,
                                const cnnlDataType_t support_type[],
                                const int &len,
                                bool &zero_element);
#endif  //  KERNELS_BINARY_OP_BINARY_OP_HOST_H_
