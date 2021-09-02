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
#ifndef INCLUDE_TYPE_H_
#define INCLUDE_TYPE_H_

#include <string>
#include "include/cnnl_core.h"

size_t getSizeOfDataType(cnnlDataType_t dtype);

std::string getNameOfDataType(cnnlDataType_t dtype);

std::string getNameOfTensorLayout(cnnlTensorLayout_t layout);
#endif  // INCLUDE_TYPE_H_
