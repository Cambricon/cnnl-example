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
#ifndef CNNL_CORE_H_
#define CNNL_CORE_H_

#include <stdint.h>
#include "cnrt.h"

#define CNNL_DIM_MAX 8

#ifndef CNNL_WIN_API
#ifdef _WIN32
#define CNNL_WIN_API __stdcall
#else
#define CNNL_WIN_API
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/******************************************************************************
 * CNNL Return Status
 ******************************************************************************/
/*! @brief Enumeration variables describing function return status.
 */
typedef enum {
  CNNL_STATUS_SUCCESS         = 0, /*!< The operation was successfully completed. */
  CNNL_STATUS_NOT_INITIALIZED = 1,
  /*!< CNNL library was not initialized properly, which is usually caused by the
       failure of calling ::cnnlCreate, ::cnnlCreateTensorDescriptor or ::cnnlSetTensorDescriptor.
       Such error is usually due to incompatible MLU device or invalid driver environment.
       Notice that ::cnnlCreate should be called prior to any other cnnl functions.*/
  CNNL_STATUS_ALLOC_FAILED = 2,
  /*!< This error occurs when the resource allocation failed, usually caused by the failure
       of cnMallocHost, probably because of the exceeded memory usage. Please make sure that
       the memory allocated previously is deallocated as much as possible.*/
  CNNL_STATUS_BAD_PARAM = 3,
  /*!< Invalid value or parameters passed to the function, including data type, layout,
       dimensions, etc.*/
  CNNL_STATUS_INTERNAL_ERROR = 4,
  /*!< Error occurred inside of the function, which may indicate an internal error or bug in
       the library. This error is usually due to the failure of cnrtMemcpyAsync.
       Please check whether the memory passed to the function was deallocated before the completion
       of the routine.*/
  CNNL_STATUS_ARCH_MISMATCH = 5,
  /*!< Invalid MLU device which was not supported by current function.*/
  CNNL_STATUS_EXECUTION_FAILED = 6,
  /*!< Error occurred when the function failed to execute on MLU device due to multiple reasons.
       You can check whether the hardware environment, driver version and other prerequisite
       libraries are correctly installed. For more information about prerequisite libraries,
       see "Cambricon CNNL User Guide".*/
  CNNL_STATUS_NOT_SUPPORTED = 7,
  /*!< Error when the requested functionality was not supported in
       this version but would be supported in the future. */
  CNNL_STATUS_NUMERICAL_OVERFLOW = 8,
  /*!< Numerical overflow occurred when executing the function,
       which is usually due to large scale or inappropriate range of value of input tensor.*/
} cnnlStatus_t;

/******************************************************************************
 * CNNL Tensor Layout
 ******************************************************************************/
/*!
 * @brief Enumeration variables describing the data layouts in CNNL.
 *
 * The data can be defined in three, four, or five dimensions.
 *
 * Take images for example, the format of the data layout can be NCHW:
 * - N: The number of images.
 * - C: The number of image channels.
 * - H: The height of images.
 * - W: The weight of images.
 *
 * Take sequence for example, the format of the data layout can be TNC:
 * - T: The timing steps of sequence.
 * - N: The batch size of sequence.
 * - C: The alphabet size of sequence.
 */
typedef enum {
  CNNL_LAYOUT_NCHW = 0,
  /*!< The data layout is in the following order: batch size, channel, height, and width. */
  CNNL_LAYOUT_NHWC = 1,
  /*!< The data layout is in the following order: batch size, height, width, and channel. */
  CNNL_LAYOUT_HWCN = 2,
  /*!< The data layout is in the following order: height, width, channel and batch size. */
  CNNL_LAYOUT_NDHWC = 3,
  /*!< The data layout is in the following order: batch size, depth, height, width, and channel.*/
  CNNL_LAYOUT_ARRAY = 4,
  /*!< The data is multi-dimensional tensor. */
  CNNL_LAYOUT_NCDHW = 5,
  /*!< The data layout is in the following order: batch size, channel, depth, height, and width.*/
  CNNL_LAYOUT_TNC = 6,
  /*!< The data layout is in the following order: timing steps, batch size, alphabet size.*/
  CNNL_LAYOUT_NTC = 7,
  /*!< The data layout is in the following order: batch size, timing steps, alphabet size.*/
  CNNL_LAYOUT_NC = 8,
  /*!< The data layout is in the following order: batch size, channel.*/
  CNNL_LAYOUT_NLC = 9,
  /*!< The data layout is in the following order: batch size, width, channel.*/
} cnnlTensorLayout_t;

/******************************************************************************
 * CNNL Data Type
 ******************************************************************************/
/*! @brief Enumeration variables describing the data types in CNNL. */
typedef enum {
  CNNL_DTYPE_INVALID = 0, /*!< The data is an invalid data type. */
  CNNL_DTYPE_HALF    = 1, /*!< The data is a 16-bit floating-point data type. */
  CNNL_DTYPE_FLOAT   = 2, /*!< The data is a 32-bit floating-point data type. */
  CNNL_DTYPE_INT8    = 3, /*!< The data is a 8-bit signed integer data type. */
  CNNL_DTYPE_INT16   = 4, /*!< The data is a 16-bit signed integer data type. */
  CNNL_DTYPE_INT31   = 5, /*!< The data is a 31-bit signed integer data type. */
  CNNL_DTYPE_INT32   = 6, /*!< The data is a 32-bit signed integer data type. */
  CNNL_DTYPE_UINT8   = 7, /*!< The data is a 8-bit unsigned integer data type. */
  CNNL_DTYPE_BOOL    = 8, /*!< The data is a BOOL data type. */
} cnnlDataType_t;

/*!
 * @brief Enumeration variables describing the options that can help choose
 *        the best suited algorithm used for implementation of the activation
 *        and accumulation operations.
 **/
typedef enum {
  CNNL_COMPUTATION_FAST = 0,
  /*!< Implementation with the fastest algorithm and lower precision.*/
  CNNL_COMPUTATION_HIGH_PRECISION = 1,
  /*!< Implementation with the high-precision algorithm regardless the performance.*/
} cnnlComputationPreference_t;

/******************************************************************************
 * CNNL Runtime Management
 ******************************************************************************/

/*!
 * @struct cnnlContext
 * @brief The \b cnnlContext is a structure describing the CNNL context.
 *
 *
 */
struct cnnlContext;
/*!
 * A pointer to ::cnnlContext struct that holds the CNNL context.
 *
 * MLU device resources cannot be accessed directly, so CNNL uses
 * handle to manage CNNL context including MLU device information
 * and queues.
 *
 * The CNNL context is created with ::cnnlCreate and the returned
 * handle should be passed to all the subsequent function calls.
 * You need to destroy the CNNL context at the end with ::cnnlDestroy.
 * For more information, see "Cambricon CNNL User Guide".
 *
 */
typedef struct cnnlContext *cnnlHandle_t;

/*! The descriptor of the collection of tensor which is used in the RNN operation, such as weight,
 * bias, etc.
 *  You need to call the ::cnnlCreateTensorSetDescriptor function to create a descriptor, and
 *  call the ::cnnlInitTensorSetMemberDescriptor to set the information about each tensor in
 *  the tensor set. If the data type of the tensor in the tensor set is in fixed-point data type,
 *  call ::cnnlInitTensorSetMemberDescriptorPositionAndScale function to set quantization
 * parameters.
 *  At last, you need to destroy the descriptor at the end with the ::cnnlDestroyTensorSetDescriptor
 *  function. */
typedef struct cnnlTensorSetStruct *cnnlTensorSetDescriptor_t;

/*!
 *  @brief Initializes the CNNL library and creates a handle \b handle to a structure
 *  that holds the CNNL library context. It allocates hardware resources on the host
 *  and device. You need to call this function before any other CNNL functions.
 *
 *  You need to call the ::cnnlDestroy function to release the resources later.
 *
 *  @param[out] handle
 *    Output. Pointer to the CNNL context that is used to manage MLU devices and
 *    queues. For detailed information, see ::cnnlHandle_t.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 */
cnnlStatus_t CNNL_WIN_API cnnlCreate(cnnlHandle_t *handle);

/*!
 *  @brief Updates the CNNL context information that holds by the \b handle. This function
 *  should be called if you call Cambriocn Driver API cnSetCtxConfigParam to set the context
 *  information. The related context information will be synchronized to CNNL with this function.
 *  For detailed information, see Cambricon Driver API Developer Guide.
 *
 *  @param[in] handle
 *    Input. Pointer to the CNNL context that is used to manage MLU devices and
 *    queues. For detailed information, see ::cnnlHandle_t.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 */
cnnlStatus_t CNNL_WIN_API cnnlUpdateContextInformation(cnnlHandle_t handle);

/*!
 *  @brief Releases the resources of the specified CNNL handle \b handle that was
 *  created by the ::cnnlCreate function.
 *  It is usually the last call to destroy the handle to the CNNL handle.
 *
 *  @param[in] handle
 *    Input. Pointer to the MLU devices that holds information to be destroyed.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlDestroy(cnnlHandle_t handle);

/*!
 *  @brief Sets the runtime queue \b queue in the handle \b handle. The queue is used to
 *  launch kernels or to synchronize to this queue.
 *
 *  Before setting a queue \b queue, you need to call the ::cnnlCreate function to initialize
 *  CNNL library, and call the cnrtCreateQueue function to create a queue \b queue.
 *
 *  @param[in] handle
 *    Input. Handle to a CNNL context that is used to manage MLU devices and
 *    queues. For detailed information, see ::cnnlHandle_t.
 *  @param[in] queue
 *    Input. The runtime queue to be set to the CNNL handle.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlSetQueue(cnnlHandle_t handle, cnrtQueue_t queue);

/*!
 *  @brief Retrieves the queue \b queue that was previously set to the handle \b handle.
 *
 *  @param[in] handle
 *    Input. Handle to a CNNL context that is used to manage MLU devices and
 *    queues. For detailed information, see ::cnnlHandle_t.
 *  @param[out] queue
 *    Output. Pointer to the queue that was previously set to the specified handle.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlGetQueue(cnnlHandle_t handle, cnrtQueue_t *queue);

/*!
 *  @brief Converts the CNNL enumerated status code to ASCIIZ static string and returns
 *  a pointer to the MLU memory that holds information about ASCIIZ static string with the status
 * name.
 *  For example, when the input argument is
 *  ::CNNL_STATUS_SUCCESS, the returned string is CNNL_STATUS_SUCCESS. When an invalid status value
 * is passed
 *  to the function, the returned string is ::CNNL_STATUS_BAD_PARAM.
 *
 *  @param[in] status
 *    Input. The CNNL enumerated status code.
 *  @return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 */
const char *cnnlGetErrorString(cnnlStatus_t status);

/******************************************************************************
 * CNNL Data Structure: Descriptor
 * The struct represent neural, weight and the neural-network layer
 ******************************************************************************/
/*! The descriptor of a tensor that holds the information including tensor
 *  layout, data type, the number of dimensions, shape and strides.
 *
 *  You need to call the ::cnnlCreateTensorDescriptor function to create a descriptor,
 *  and call the ::cnnlSetTensorDescriptor function or the ::cnnlSetTensorDescriptorEx
 *  function to set the tensor information to the descriptor. Also, you need to destroy
 *  the CNNL context at the end with the ::cnnlDestroyTensorDescriptor function.
 */
typedef struct cnnlTensorStruct *cnnlTensorDescriptor_t;

/*!
 *  @brief Creates a tensor descriptor pointed by \b desc that holds the dimensions, data type,
 *  and layout of input tensor. If the input tensor is in fixed-point data type,
 *  the ::cnnlSetTensorDescriptorPositionAndScale function or the ::cnnlSetTensorDescriptorPosition
 *  function needs to be called to set quantization parameters.
 *
 *  The ::cnnlDestroyTensorDescriptor function needs to be called to destroy the
 *  tensor descriptor later.
 *
 *  @param[in] desc
 *    Input. Pointer to the struct that holds information about the tensor descriptor.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlCreateTensorDescriptor(cnnlTensorDescriptor_t *desc);

/*!
 *  @brief Initializes the tensor descriptor pointed by \b desc that is previously created
 *  with the ::cnnlCreateTensorDescriptor function, and sets the information about
 *  the dimensions, data type, and layout of the input tensor.
 *
 *  If ::cnnlSetTensorDescriptor is called, you do not need to specify the strides of all
 *  dimensions. The strides are inferred by parameters passed to this function. Also, the data
 *  will be treated as contiguous in memory with no padding between dimensions. To specify the
 *  strides of all dimensions, you can call ::cnnlSetTensorDescriptorEx. But the data might not
 *  be treated as contiguous in memory.
 *
 *  @param[in] desc
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::cnnlTensorDescriptor_t.
 *  @param[in] layout
 *    Input. The layout of the input tensor. For detailed information, see ::cnnlTensorLayout_t.
 *  @param[in] dtype
 *    Input. The data type of the input tensor. For detailed information, see ::cnnlDataType_t.
 *  @param[in] dimNb
 *    Input. The number of dimensions in the input tensor of the initialized operation.
 *  @param[in] dimSize
 *    Input. An array that contains the size of the tensor for each dimension.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - dimSize[0] represents the highest dimension, dimSize[DIM_MAX - 1] represents
 *    the lowest dimension, and DIM_MAX represents the number of dimensions in the input tensor.
 *  - This function cannot be called continuously. You need to call ::cnnlResetTensorDescriptor
 *    before calling another ::cnnlSetTensorDescriptor to avoid memory leaks.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlSetTensorDescriptor(cnnlTensorDescriptor_t desc,
                                                  cnnlTensorLayout_t layout,
                                                  cnnlDataType_t dtype,
                                                  int dimNb,
                                                  const int dimSize[]);

/*!
 *  @brief Resets the tensor descriptor pointed by \b desc that is previously created
 *  with the ::cnnlCreateTensorDescriptor function.
 *
 *  If ::cnnlResetTensorDescriptor is called, all the information about the tensor will be reset to
 *  initial value, which means layout is CNNL_LAYOUT_ARRAY, dtype is CNNL_DTYPE_FLOAT, dimsNb is 0,
 *  and dimSize points to an \b CNNL_DIM_MAX-dimension array.
 *
 *  @param[in] desc
 *    Input. The descriptor of the tensor. For detailed information, see ::cnnlTensorDescriptor_t.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - This function is used to avoid memory leaks when more than one ::cnnlSetTensorDescriptor
 *    function is called. You should call this function before calling another
 *    ::cnnlSetTensorDescriptor
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlResetTensorDescriptor(cnnlTensorDescriptor_t desc);

/*!
 *  @brief Retrieves a tensor descriptor \b desc that is previously created with the
 *  ::cnnlCreateTensorDescriptor function, and sets the information about the dimensions,
 *  data type, and layout of input tensor.
 *
 *  @param[in] desc
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::cnnlTensorDescriptor_t.
 *  @param[out] layout
 *    Output. Pointer to the host memory that holds information about the layout of the input
 * tensor.
 *  For detailed information, see ::cnnlTensorLayout_t.
 *  @param[out] dtype
 *    Output. Pointer to the host memory that holds information about the data type of the input
 * tensor.
 *  For detailed information, see ::cnnlDataType_t.
 *  @param[out] dimNb
 *    Output. Pointer to the host memory that holds information about the dimension of input tensor.
 *  @param[out] dimSize
 *    Output. An array that contains the size of the tensor for each dimension.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents the lowest
 *    dimension.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlGetTensorDescriptor(const cnnlTensorDescriptor_t desc,
                                                  cnnlTensorLayout_t *layout,
                                                  cnnlDataType_t *dtype,
                                                  int *dimNb,
                                                  int dimSize[]);

/*!
 *  @brief Retrieves the number of elements according to the input descriptor \b desc. You
 *  need to call the ::cnnlSetTensorDescriptor function first to create a tensor descriptor
 *  before calling this function.
 *
 *  @param[in] desc
 *    Input. The descriptor of input tensor. For detailed information,
 *    see ::cnnlTensorDescriptor_t.
 *  @return
 *  - ::CNNL_STATUS_SUCCESS
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
     @verbatim
      cnnlTensorDescriptor_t input_desc;
      cnnlCreateTensorDescriptor(&input_desc);
      cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_ARRAY,CNNL_DTYPE_FLOAT, 2,{2, 3});
      size_t nums=cnnlGetTensorElementNum(input_desc);  // nums = 6

      input one array by 2 * 3
      input: [[1,2,3],[4,5,6]]
      output: 6
     @endverbatim
 */
size_t CNNL_WIN_API cnnlGetTensorElementNum(const cnnlTensorDescriptor_t desc);

/*!
 *  @brief Destroies a tensor descriptor that was created by
 *         ::cnnlCreateTensorDescriptor.
 *
 *  @param[in] desc
 *    Input. A tensor descriptor created by ::cnnlCreateTensorDescriptor.
 *  @par Return
 *  - ::CNNL_STATUS_SUCCESS, ::CNNL_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
cnnlStatus_t CNNL_WIN_API cnnlDestroyTensorDescriptor(cnnlTensorDescriptor_t desc);

#if defined(__cplusplus)
}
#endif

#endif  // CNNL_CORE_H_
