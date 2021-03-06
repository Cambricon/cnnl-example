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
#ifndef KERNELS_ABS_ABS_H_
#define KERNELS_ABS_ABS_H_

#include "kernels/unary_op/unary_op_3pipeline.h"
#include "kernels/unary_op/unary_op_5pipeline.h"

// declare abs 3stage pipline kernel, only fast mode
UNARY_OP_KERNEL_3PIPELINE_DECLARE(Abs, float, Fast);
UNARY_OP_KERNEL_3PIPELINE_DECLARE(Abs, half, Fast);

// declare abs 5stage pipelinekerneol, only fast mode
UNARY_OP_KERNEL_5PIPELINE_DECLARE(Abs, float, Fast);
UNARY_OP_KERNEL_5PIPELINE_DECLARE(Abs, half, Fast);

#endif  // KERNELS_ABS_ABS_H_
