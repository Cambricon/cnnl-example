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
#ifndef INCLUDE_MACROS_H_
#define INCLUDE_MACROS_H_

#if !defined(PLATFORM_POSIX)
#if defined(_WIN32)
#define PLATFORM_WINDOWS
#else
#define PLATFORM_POSIX
#endif
#endif  // !defined(PLATFORM_POSIX)

// Compiler attributes
#if defined(__GNUC__)

// Compiler supports GCC-style attributes
#define CNNL_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define CNNL_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define CNNL_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define CNNL_ATTRIBUTE_UNUSED __attribute__((unused))
#define CNNL_ATTRIBUTE_COLD __attribute__((cold))
#define CNNL_ATTRIBUTE_WEAK __attribute__((weak))
#define CNNL_PACKED __attribute__((packed))
#define CNNL_MUST_USE_RESULT __attribute__((warn_unused_result))
#define CNNL_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define CNNL_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define CNNL_ATTRIBUTE_NORETURN __declspec(noreturn)
#define CNNL_ATTRIBUTE_ALWAYS_INLINE __forceinline
#define CNNL_ATTRIBUTE_NOINLINE
#define CNNL_ATTRIBUTE_UNUSED
#define CNNL_ATTRIBUTE_COLD
#define CNNL_ATTRIBUTE_WEAK
#define CNNL_MUST_USE_RESULT
#define CNNL_PACKED
#define CNNL_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define CNNL_SCANF_ATTRIBUTE(string_index, first_to_check)

#else  // defined(__GNUC__)

// Non-GCC equivalents
#define CNNL_ATTRIBUTE_NORETURN
#define CNNL_ATTRIBUTE_ALWAYS_INLINE
#define CNNL_ATTRIBUTE_NOINLINE
#define CNNL_ATTRIBUTE_UNUSED
#define CNNL_ATTRIBUTE_COLD
#define CNNL_ATTRIBUTE_WEAK
#define CNNL_MUST_USE_RESULT
#define CNNL_PACKED
#define CNNL_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define CNNL_SCANF_ATTRIBUTE(string_index, first_to_check)

#endif  // defined(__GNUC__)

#ifdef __has_builtin
#define CNNL_HAS_BUILTIN(x) __has_builtin(x)
#else
#define CNNL_HAS_BUILTIN(x) 0
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#if (CNNL_HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3))
#define CNNL_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define CNNL_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define CNNL_PREDICT_FALSE(x) (x)
#define CNNL_PREDICT_TRUE(x) (x)
#endif

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#define THRESHOLD_MSE (1e-5)
#define THRESHOLD_DIFF1 (0.003)

#endif  // INCLUDE_MACROS_H_
