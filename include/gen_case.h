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
#ifndef INCLUDE_GEN_CASE_H_
#define INCLUDE_GEN_CASE_H_

#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include "include/cnnl_core.h"
#include "include/tensor.h"
#include "include/type.h"
#include "include/logging.h"

#define CNNL_GEN_CASE_ON cnnl::gen_case::isGenCaseOn()
#define GEN_CASE_START(op_name, op_type) \
  std::string gen_case_file_name = cnnl::gen_case::genCaseStart(op_name, op_type)
#define GEN_CASE_DATA(is_input, id, data, data_desc, upper_bound, lower_bound)                 \
  cnnl::gen_case::genCaseData(&gen_case_file_name, is_input, id, data, data_desc, upper_bound, \
                              lower_bound)
#define GEN_CASE_DATA_UNFOLD(is_input, id, data, dim, dims, dtype, layout, upper_bound,          \
                             lower_bound)                                                        \
  cnnl::gen_case::genCaseData(&gen_case_file_name, is_input, id, data, dim, dims, dtype, layout, \
                              upper_bound, lower_bound)
#define GEN_CASE_OP_PARAM_SINGLE(flag, op_name, param_name, value) \
  cnnl::gen_case::genCaseOpParam(flag, &gen_case_file_name, op_name, param_name, value)
#define GEN_CASE_OP_PARAM_ARRAY(flag, op_name, param_name, value, num) \
  cnnl::gen_case::genCaseOpParam(flag, &gen_case_file_name, op_name, param_name, value, num)
#define GEN_CASE_TEST_PARAM(is_diff1, is_diff2, is_diff3, diff1_threshold, diff2_threshold, \
                            diff3_threshold)                                                \
  cnnl::gen_case::genCaseTestParam(&gen_case_file_name, is_diff1, is_diff2, is_diff3,       \
                                   diff1_threshold, diff2_threshold, diff3_threshold)

namespace cnnl {
namespace gen_case {

bool isGenCaseOn();
void genCaseModeSet(int mode);
void genCaseModeSet(std::string mode);
float cvtHalfToFloat(int16_t src);
void saveDataToFile(std::string gen_case_file_name,
                    void *data,
                    cnnlDataType_t dtype,
                    int32_t count);
std::string genCaseStart(std::string op_name, std::string op_type);
void genCaseData(std::string *gen_case_file_name,
                 bool is_input,
                 std::string id,
                 const void *device_data,
                 const int dim,
                 int *dims,
                 cnnlDataType_t dtype,
                 cnnlTensorLayout_t layout,
                 const float upper_bound,
                 const float lower_bound);
void genCaseData(std::string *gen_case_file_name,
                 bool is_input,
                 std::string id,
                 const void *device_data,
                 const int dim,
                 std::vector<int> dims,
                 cnnlDataType_t dtype,
                 cnnlTensorLayout_t layout,
                 const float upper_bound,
                 const float lower_bound);
void genCaseData(std::string *gen_case_file_name,
                 bool is_input,
                 std::string id,
                 const void *device_data,
                 const cnnlTensorDescriptor_t data_desc,
                 const float upper_bound,
                 const float lower_bound);
void genCaseOpParam(const int flag,
                    std::string *gen_case_file_name,
                    std::string op_name,
                    std::string param_name,
                    const float value);
void genCaseOpParam(const int flag,
                    std::string *gen_case_file_name,
                    std::string op_name,
                    std::string param_name,
                    std::string value);
void genCaseOpParam(const int flag,
                    std::string *gen_case_file_name,
                    std::string op_name,
                    std::string param_name,
                    const int *value,
                    const int num);
void genCaseOpParam(const int flag,
                    std::string *gen_case_file_name,
                    std::string op_name,
                    std::string param_name,
                    const float *value,
                    const int num);
void genCaseTestParam(std::string *gen_case_file_name,
                      bool is_diff1,
                      bool is_diff2,
                      bool is_diff3,
                      const float diff1_threshold,
                      const float diff2_threshold,
                      const float diff3_threshold);
int getIntEnvVar(const std::string &str, int default_para);
bool getBoolEnvVar(const std::string &str, bool default_para);
std::string getStringEnvVar(const std::string &str, std::string default_para);
bool getBoolOpName(const std::string str, const std::string op_name);
}  // namespace gen_case
}  // namespace cnnl
#endif  // INCLUDE_GEN_CASE_H_
