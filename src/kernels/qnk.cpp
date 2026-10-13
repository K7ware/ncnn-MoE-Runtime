#include "qnk.h"

#include "mxfp4.h"
#include "qnk_msvc.h"
#include "activationbuffer.h"
#include "ops.h"
#include "vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace ncnn {
namespace moe {

static float half_to_float(uint16_t value) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = sign;
    if (exponent == 0)
    {
        if (mantissa != 0)
        {
            uint32_t normalized = mantissa;
            int32_t exponent_shift = 0;
            while ((normalized & 0x0400u) == 0)
            {
                normalized <<= 1;
                --exponent_shift;
            }
            normalized &= 0x03ffu;
            bits |= static_cast<uint32_t>(127 - 15 + 1 + exponent_shift) << 23;
            bits |= normalized << 13;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits |= 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits |= (exponent + (127 - 15)) << 23;
        bits |= mantissa << 13;
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t load_u16(const uint8_t* source) noexcept
{
    uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

static float load_f16(const uint8_t* source) noexcept
{
    return half_to_float(load_u16(source));
}

struct QnKOutput
{
    float* output;
    const float* input;
#if defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t sum0;
    float32x4_t sum1;
    float32x4_t sum2;
    float32x4_t sum3;
#endif
};

static void get_scale_min_k4(int index,
                             const uint8_t* scales,
                             uint8_t& scale,
                             uint8_t& minimum) noexcept
{
    if (index < 4)
    {
        scale = scales[index] & 63u;
        minimum = scales[index + 4] & 63u;
    }
    else
    {
        scale = static_cast<uint8_t>((scales[index + 4] & 0x0fu) | ((scales[index - 4] >> 6) << 4));
        minimum = static_cast<uint8_t>((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
static inline void qnk_i8_to_f32(int8x8_t values,
                                 float scale,
                                 float minimum,
                                 bool affine,
                                 QnKOutput& output,
                                 uint32_t index) noexcept
{
    const int16x8_t wide = vmovl_s8(values);
    float32x4_t low = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide))), scale);
    float32x4_t high = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide))), scale);
    if (affine)
    {
        const float32x4_t min_values = vdupq_n_f32(minimum);
        low = vsubq_f32(low, min_values);
        high = vsubq_f32(high, min_values);
    }
    if (!output.input)
    {
        vst1q_f32(output.output + index, low);
        vst1q_f32(output.output + index + 4, high);
    }
    else if ((index & 8u) == 0)
    {
        output.sum0 = vfmaq_f32(output.sum0, low, vld1q_f32(output.input + index));
        output.sum1 = vfmaq_f32(output.sum1, high, vld1q_f32(output.input + index + 4));
    }
    else
    {
        output.sum2 = vfmaq_f32(output.sum2, low, vld1q_f32(output.input + index));
        output.sum3 = vfmaq_f32(output.sum3, high, vld1q_f32(output.input + index + 4));
    }
}
#endif

static void dequantize_q2(const uint8_t* block, QnKOutput& output) noexcept
{
    const float d = load_f16(block + 80);
    const float minimum = load_f16(block + 82);
    const uint8_t* scales = block;
    const uint8_t* q = block + 16;
    int scale_index = 0;
    for (uint32_t n = 0; n < qnk_block_elements; n += 128)
    {
        int shift = 0;
        for (int group = 0; group < 4; ++group)
        {
            for (int half = 0; half < 2; ++half)
            {
                const uint8_t scale_min = scales[scale_index++];
                const float dl = d * (scale_min & 0x0fu);
                const float ml = minimum * (scale_min >> 4);
                const uint8_t* source = q + n / 4 + half * 16;
                const uint32_t index = n + group * 32 + half * 16;
#if defined(__aarch64__) || defined(_M_ARM64)
                for (int lane = 0; lane < 16; lane += 8)
                {
                    const int8x8_t shift_count = vdup_n_s8(static_cast<int8_t>(-shift));
                    const uint8x8_t values = vand_u8(vshl_u8(vld1_u8(source + lane), shift_count),
                                                     vdup_n_u8(3));
                    qnk_i8_to_f32(vreinterpret_s8_u8(values), dl, ml, true,
                                  output, index + lane);
                }
#else
                for (int lane = 0; lane < 16; ++lane)
                    output.output[index + lane] = dl * static_cast<float>((source[lane] >> shift) & 3u) - ml;
#endif
            }
            shift += 2;
        }
    }
}

static void dequantize_q3(const uint8_t* block, QnKOutput& output) noexcept
{
    constexpr uint32_t kmask1 = 0x03030303u;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t auxiliary[4] = {};
    std::memcpy(auxiliary, block + 96, 12);
    const uint32_t temporary = auxiliary[2];
    auxiliary[2] = ((auxiliary[0] >> 4) & kmask2) | (((temporary >> 4) & kmask1) << 4);
    auxiliary[3] = ((auxiliary[1] >> 4) & kmask2) | (((temporary >> 6) & kmask1) << 4);
    auxiliary[0] = (auxiliary[0] & kmask2) | (((temporary >> 0) & kmask1) << 4);
    auxiliary[1] = (auxiliary[1] & kmask2) | (((temporary >> 2) & kmask1) << 4);
    int8_t scales[16] = {};
    std::memcpy(scales, auxiliary, sizeof(scales));
    const float d = load_f16(block + 108);
    const uint8_t* q = block + 32;
    const uint8_t* high_bits = block;
    uint8_t mask = 1;
    int scale_index = 0;
    for (uint32_t n = 0; n < qnk_block_elements; n += 128)
    {
        int shift = 0;
        for (int group = 0; group < 4; ++group)
        {
            for (int half = 0; half < 2; ++half)
            {
                const float scale = d * static_cast<float>(scales[scale_index++] - 32);
#if defined(__aarch64__) || defined(_M_ARM64)
                for (int lane = 0; lane < 16; lane += 8)
                {
                    const uint8_t* source = q + n / 4 + half * 16 + lane;
                    const uint8x8_t values = vand_u8(vshl_u8(vld1_u8(source),
                                                             vdup_n_s8(static_cast<int8_t>(-shift))),
                                                     vdup_n_u8(3));
                    const uint8x8_t bits = vand_u8(vld1_u8(high_bits + half * 16 + lane),
                                                   vdup_n_u8(mask));
                    const uint8x8_t high = vand_u8(vceq_u8(bits, vdup_n_u8(0)),
                                                   vdup_n_u8(4));
                    qnk_i8_to_f32(vreinterpret_s8_u8(vsub_u8(values, high)),
                                  scale, 0.0f, false,
                                  output, n + group * 32 + half * 16 + lane);
                }
#else
                for (int lane = 0; lane < 16; ++lane)
                {
                    const int high = (high_bits[half * 16 + lane] & mask) != 0 ? 0 : 4;
                    const int value = static_cast<int>((q[n / 4 + half * 16 + lane] >> shift) & 3u) - high;
                    output.output[n + group * 32 + half * 16 + lane] = scale * value;
                }
#endif
            }
            shift += 2;
            mask = static_cast<uint8_t>(mask << 1);
        }
    }
}

static void dequantize_q4(const uint8_t* block, QnKOutput& output) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* q = block + 16;
    int scale_index = 0;
    for (uint32_t n = 0; n < qnk_block_elements; n += 64)
    {
        const uint8_t* source = q + n / 2;
        for (int half = 0; half < 2; ++half)
        {
            uint8_t scale_value = 0;
            uint8_t min_value = 0;
            get_scale_min_k4(scale_index++, scales, scale_value, min_value);
            const float scale = d * scale_value;
            const float minimum_value = minimum * min_value;
            const uint32_t index = n + half * 32;
#if defined(__aarch64__) || defined(_M_ARM64)
            for (int lane = 0; lane < 32; lane += 8)
            {
                const uint8x8_t packed = vld1_u8(source + lane);
                const uint8x8_t values = half == 0 ? vand_u8(packed, vdup_n_u8(0x0fu)) : vshr_n_u8(packed, 4);
                qnk_i8_to_f32(vreinterpret_s8_u8(values), scale, minimum_value, true, output, index + lane);
            }
#else
            for (int lane = 0; lane < 32; ++lane)
            {
                const uint8_t packed = source[lane];
                const uint8_t value = half == 0 ? packed & 0x0fu : packed >> 4;
                output.output[index + lane] = scale * static_cast<float>(value) - minimum_value;
            }
#endif
        }
    }
}

static void dequantize_q5(const uint8_t* block, QnKOutput& output) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* high_bits = block + 16;
    const uint8_t* low_bits = block + 48;
    int scale_index = 0;
    uint8_t first_mask = 1;
    uint8_t second_mask = 2;
    for (uint32_t n = 0; n < qnk_block_elements; n += 64)
    {
        for (int half = 0; half < 2; ++half)
        {
            uint8_t scale_value = 0;
            uint8_t min_value = 0;
            get_scale_min_k4(scale_index++, scales, scale_value, min_value);
            const float scale = d * scale_value;
            const float minimum_value = minimum * min_value;
            const uint8_t mask = half == 0 ? first_mask : second_mask;
            const uint32_t index = n + half * 32;
#if defined(__aarch64__) || defined(_M_ARM64)
            for (int lane = 0; lane < 32; lane += 8)
            {
                const uint8x8_t low = vld1_u8(low_bits + n / 2 + lane);
                const uint8x8_t high = vld1_u8(high_bits + lane);
                const uint8x8_t unpacked = half == 0 ? vand_u8(low, vdup_n_u8(0x0fu)) : vshr_n_u8(low, 4);
                const uint8x8_t high_value = vand_u8(vcgt_u8(vand_u8(high, vdup_n_u8(mask)),
                                                             vdup_n_u8(0)),
                                                     vdup_n_u8(16));
                qnk_i8_to_f32(vreinterpret_s8_u8(vadd_u8(unpacked, high_value)),
                              scale, minimum_value, true, output, index + lane);
            }
#else
            for (int lane = 0; lane < 32; ++lane)
            {
                const uint8_t low = low_bits[n / 2 + lane];
                const int high = (high_bits[lane] & mask) != 0 ? 16 : 0;
                const uint8_t value = half == 0 ? low & 0x0fu : low >> 4;
                output.output[index + lane] = scale * static_cast<float>(value + high) - minimum_value;
            }
#endif
        }
        first_mask = static_cast<uint8_t>(first_mask << 2);
        second_mask = static_cast<uint8_t>(second_mask << 2);
    }
}

// Keep the decoder in the caller so fused accumulators remain local.
#if defined(_MSC_VER) && defined(_M_ARM64)
static __forceinline void dequantize_q6(const uint8_t* block, QnKOutput& output) noexcept
#else
#if defined(__aarch64__)
__attribute__((always_inline))
#endif
static inline void dequantize_q6(const uint8_t* block, QnKOutput& output) noexcept
#endif
{
    const float d = load_f16(block + 208);
    const uint8_t* low_bits = block;
    const uint8_t* high_bits = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    for (uint32_t n = 0; n < qnk_block_elements; n += 128)
    {
        const uint8_t* low = low_bits + n / 2;
        const uint8_t* high = high_bits + n / 4;
        const int scale_base = static_cast<int>(n / 16);
#if defined(__aarch64__) || defined(_M_ARM64)
        for (int quarter = 0; quarter < 4; ++quarter)
        {
            for (int lane = 0; lane < 32; lane += 8)
            {
                const uint8x8_t packed_low = vld1_u8(low + (quarter & 1) * 32 + lane);
                const uint8x8_t low_value = quarter < 2
                                                ? vand_u8(packed_low, vdup_n_u8(0x0fu))
                                                : vshr_n_u8(packed_low, 4);
                const uint8x8_t high_value = vand_u8(vshl_u8(vld1_u8(high + lane),
                                                             vdup_n_s8(static_cast<int8_t>(-2 * quarter))),
                                                     vdup_n_u8(3));
                const uint8x8_t value = vsub_u8(vorr_u8(low_value, vshl_n_u8(high_value, 4)), vdup_n_u8(32));
                const float scale = d * scales[scale_base + 2 * quarter + lane / 16];
                qnk_i8_to_f32(vreinterpret_s8_u8(value), scale, 0.0f, false, output,
                              n + quarter * 32 + lane);
            }
        }
#else
        for (int lane = 0; lane < 32; ++lane)
        {
            const int q1 = static_cast<int>((low[lane] & 0x0fu) | (((high[lane] >> 0) & 3u) << 4)) - 32;
            const int q2 = static_cast<int>((low[lane + 32] & 0x0fu) | (((high[lane] >> 2) & 3u) << 4)) - 32;
            const int q3 = static_cast<int>((low[lane] >> 4) | (((high[lane] >> 4) & 3u) << 4)) - 32;
            const int q4 = static_cast<int>((low[lane + 32] >> 4) | (((high[lane] >> 6) & 3u) << 4)) - 32;
            output.output[n + lane] = d * scales[scale_base + lane / 16] * q1;
            output.output[n + lane + 32] = d * scales[scale_base + lane / 16 + 2] * q2;
            output.output[n + lane + 64] = d * scales[scale_base + lane / 16 + 4] * q3;
            output.output[n + lane + 96] = d * scales[scale_base + lane / 16 + 6] * q4;
        }
#endif
    }
}

static void dequantize_q8(const uint8_t* block, QnKOutput& output) noexcept
{
    float scale = 0.0f;
    std::memcpy(&scale, block, sizeof(scale));
    const int8_t* values = reinterpret_cast<const int8_t*>(block + 4);
#if defined(__aarch64__) || defined(_M_ARM64)
    for (uint32_t index = 0; index < qnk_block_elements; index += 16)
    {
        const int8x16_t packed = vld1q_s8(values + index);
        qnk_i8_to_f32(vget_low_s8(packed), scale, 0.0f, false, output, index);
        qnk_i8_to_f32(vget_high_s8(packed), scale, 0.0f, false, output, index + 8);
    }
#else
    for (uint32_t index = 0; index < qnk_block_elements; ++index)
        output.output[index] = scale * static_cast<float>(values[index]);
#endif
}

static void qnk_gemm(const uint8_t* raw,
                     const QnKPack* packed,
                     DType dtype,
                     size_t rows,
                     uint32_t columns,
                     const float* input,
                     size_t input_stride,
                     size_t token_count,
                     float* output,
                     size_t output_stride) noexcept
{
    const uint32_t block_count = columns / qnk_block_elements;
    const size_t block_bytes = qnk_block_bytes(dtype);
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(_OPENMP)
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const size_t task_count = token_count > 1 ? (rows + 7) / 8 : rows;
    const int team_size = static_cast<int>(std::min<size_t>(task_count,
                                                            static_cast<size_t>(cpu_linear_team_size(operation_count, dtype))));
#endif
    if (token_count == 1)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
#endif
        for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row)
        {
            float sum = 0.0f;
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* encoded = packed
                                             ? qnk_packed_block(*packed, static_cast<size_t>(row), block)
                                             : raw + ((static_cast<size_t>(row) * block_count + block) * block_bytes);
                sum += qnk_dot_block(dtype,
                                     encoded,
                                     input + static_cast<size_t>(block) * qnk_block_elements);
            }
            output[row] = sum;
        }
        return;
    }
    if (token_count > 1)
    {
        const size_t tile_count = (rows + 7) / 8;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
#endif
        for (int64_t tile = 0; tile < static_cast<int64_t>(tile_count); ++tile)
        {
            const size_t first_row = static_cast<size_t>(tile) * 8;
            const uint32_t output_count = static_cast<uint32_t>(std::min<size_t>(8, rows - first_row));
            alignas(16) float decoded[8 * qnk_block_elements];
            for (uint32_t block = 0; block < block_count; ++block)
            {
                for (uint32_t row = 0; row < output_count; ++row)
                {
                    const size_t weight_row = first_row + row;
                    const uint8_t* encoded = packed
                                                 ? qnk_packed_block(*packed, weight_row, block)
                                                 : raw + ((weight_row * block_count + block) * block_bytes);
                    qnk_dequantize_block(dtype, encoded, decoded + static_cast<size_t>(row) * qnk_block_elements);
                }
                for (size_t first_token = 0; first_token < token_count; first_token += 4)
                {
                    const uint32_t input_count = static_cast<uint32_t>(std::min<size_t>(4, token_count - first_token));
                    const float* input_tile = input + first_token * input_stride
                                              + static_cast<size_t>(block) * qnk_block_elements;
                    float* destination = output + first_token * output_stride + first_row;
                    float partial[4 * 8];
                    float_gemm_4x8(decoded,
                                   qnk_block_elements,
                                   input_tile,
                                   input_stride,
                                   qnk_block_elements,
                                   output_count,
                                   input_count,
                                   block == 0 ? destination : partial,
                                   block == 0 ? output_stride : 8);
                    if (block != 0)
                    {
                        for (uint32_t token = 0; token < input_count; ++token)
                        {
                            float* token_output = output + (first_token + token) * output_stride + first_row;
                            const float* values = partial + token * 8;
                            for (uint32_t row = 0; row < output_count; ++row)
                                token_output[row] += values[row];
                        }
                    }
                }
            }
        }
        return;
    }
#endif
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t token = 0; token < token_count; ++token)
        {
            const float* token_input = input + token * input_stride;
            float sum = 0.0f;
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* encoded = packed
                                             ? qnk_packed_block(*packed, row, block)
                                             : raw + ((row * static_cast<size_t>(block_count) + block) * block_bytes);
                sum += qnk_dot_block(dtype,
                                     encoded,
                                     token_input + static_cast<size_t>(block) * qnk_block_elements);
            }
            output[token * output_stride + row] = sum;
        }
    }
}

static std::shared_ptr<const QnKPack> get_qnk_packed_weights(const TensorData& matrix,
                                                             std::shared_ptr<const QnKPack>* sidecar,
                                                             size_t rows,
                                                             uint32_t columns) noexcept
{
    static std::mutex build_locks[64];
    const std::span<const uint8_t> raw = matrix.qnk_values();
    const uintptr_t storage_key = reinterpret_cast<uintptr_t>(raw.data());
    std::lock_guard<std::mutex> lock(build_locks[(storage_key >> 6) & 63u]);
    std::shared_ptr<const QnKPack> cached = sidecar ? *sidecar : nullptr;
    if (cached && cached->valid() && cached->dtype == matrix.dtype && cached->rows == rows && cached->columns == columns)
        return cached;

    auto packed = std::make_shared<QnKPack>();
    if (!qnk_pack_weights(raw.data(), raw.size(), matrix.dtype, rows, columns, *packed))
        return {};
    if (sidecar)
        *sidecar = packed;
    return packed;
}

size_t qnk_block_bytes(DType dtype) noexcept
{
    switch (dtype)
    {
    case DType::Q2K: return 84;
    case DType::Q3K: return 110;
    case DType::Q4K: return 144;
    case DType::Q5K: return 176;
    case DType::Q6K: return 210;
    case DType::Q8K: return 292;
    default: return 0;
    }
}

uint64_t qnk_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept
{
    if (!qnk_shape_supported(dtype, rows, columns))
        return 0;
    const uint64_t row_blocks = static_cast<uint64_t>(columns / qnk_block_elements);
    const uint64_t bytes_per_block = qnk_block_bytes(dtype);
    if (row_blocks != 0 && bytes_per_block > std::numeric_limits<uint64_t>::max() / row_blocks)
        return 0;
    const uint64_t row_bytes = row_blocks * bytes_per_block;
    if (rows != 0 && row_bytes > std::numeric_limits<uint64_t>::max() / rows)
        return 0;
    return row_bytes * rows;
}

uint64_t qnk_packed_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept
{
    if (!qnk_shape_supported(dtype, rows, columns)
        || rows > std::numeric_limits<size_t>::max() - 7)
    {
        return 0;
    }
    const uint64_t padded_rows = (rows + 7) / 8 * 8;
    const uint64_t block_count = columns / qnk_block_elements;
    const uint64_t block_bytes = qnk_block_bytes(dtype);
    if (padded_rows > std::numeric_limits<uint64_t>::max() / block_count
        || padded_rows * block_count > std::numeric_limits<uint64_t>::max() / block_bytes)
    {
        return 0;
    }
    return padded_rows * block_count * block_bytes;
}

bool qnk_shape_supported(DType dtype, size_t rows, uint32_t columns) noexcept
{
    return is_qnk_dtype(dtype) && rows != 0 && columns != 0 && columns % qnk_block_elements == 0
           && qnk_block_bytes(dtype) != 0;
}

static inline float forward_qnk(DType dtype,
                                const uint8_t* block,
                                const float* input,
                                float* output) noexcept
{
    QnKOutput local;
    local.output = output;
    local.input = input;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (local.input)
    {
        local.sum0 = vdupq_n_f32(0.0f);
        local.sum1 = vdupq_n_f32(0.0f);
        local.sum2 = vdupq_n_f32(0.0f);
        local.sum3 = vdupq_n_f32(0.0f);
    }
#endif
    switch (dtype)
    {
    case DType::Q2K: dequantize_q2(block, local); break;
    case DType::Q3K: dequantize_q3(block, local); break;
    case DType::Q4K: dequantize_q4(block, local); break;
    case DType::Q5K: dequantize_q5(block, local); break;
    case DType::Q6K: dequantize_q6(block, local); break;
    case DType::Q8K: dequantize_q8(block, local); break;
    default:
        if (output)
            std::fill_n(output, qnk_block_elements, 0.0f);
        return 0.0f;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    if (local.input)
    {
        local.sum0 = vaddq_f32(vaddq_f32(local.sum0, local.sum1),
                               vaddq_f32(local.sum2, local.sum3));
        return vaddvq_f32(local.sum0);
    }
#endif
    return 0.0f;
}

void qnk_dequantize_block(DType dtype, const uint8_t* block, float* output) noexcept
{
    if (!block || !output)
        return;
    if (dtype == DType::Q6K)
    {
        QnKOutput destination;
        destination.output = output;
        destination.input = nullptr;
        dequantize_q6(block, destination);
        return;
    }
    forward_qnk(dtype, block, nullptr, output);
}

float qnk_dot_block(DType dtype, const uint8_t* block, const float* input) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (is_qnk_dtype(dtype))
    {
        if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
            return msvc_avx512_qnk_dot_block(dtype, block, input);
        if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
            return msvc_avx2_qnk_dot_block(dtype, block, input);
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (block && input && (dtype == DType::Q2K || dtype == DType::Q3K || dtype == DType::Q4K || dtype == DType::Q5K || dtype == DType::Q6K || dtype == DType::Q8K))
        return forward_qnk(dtype, block, input, nullptr);
#endif
    alignas(64) float decoded[qnk_block_elements];
    qnk_dequantize_block(dtype, block, decoded);
#if defined(__aarch64__) || defined(_M_ARM64)
    return float_dot(decoded, input, qnk_block_elements);
#else
    float sum = 0.0f;
    for (uint32_t index = 0; index < qnk_block_elements; ++index)
        sum += decoded[index] * input[index];
    return sum;
#endif
}

void qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept
{
    if (!source || !output || columns == 0 || columns % qnk_block_elements != 0)
        return;
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
    {
        msvc_avx512_qnk_q8k_quantize(source, output, columns);
        return;
    }
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
    {
        msvc_avx2_qnk_q8k_quantize(source, output, columns);
        return;
    }
#endif
    const uint32_t block_count = columns / qnk_block_elements;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const float* values = source + static_cast<size_t>(block_index) * qnk_block_elements;
        uint8_t* encoded = output + static_cast<size_t>(block_index) * qnk_block_bytes(DType::Q8K);
        float maximum = 0.0f;
        float signed_maximum = 0.0f;
#if defined(__aarch64__) || defined(_M_ARM64)
        bool finite = true;
#endif
        for (uint32_t index = 0; index < qnk_block_elements; ++index)
        {
            const float value = values[index];
#if defined(__aarch64__) || defined(_M_ARM64)
            finite = finite && std::isfinite(value);
#endif
            const float absolute = std::fabs(value);
            if (absolute > maximum)
            {
                maximum = absolute;
                signed_maximum = value;
            }
        }
        if (maximum == 0.0f)
        {
            std::memset(encoded, 0, qnk_block_bytes(DType::Q8K));
            continue;
        }
        const float inverse_scale = -127.0f / signed_maximum;
        const float scale = 1.0f / inverse_scale;
        std::memcpy(encoded, &scale, sizeof(scale));
        int8_t* quantized = reinterpret_cast<int8_t*>(encoded + 4);
#if defined(__aarch64__) || defined(_M_ARM64)
        if (finite && std::isfinite(inverse_scale))
        {
            const int32x4_t minimum_value = vdupq_n_s32(-127);
            const int32x4_t maximum_value = vdupq_n_s32(127);
            for (uint32_t index = 0; index < qnk_block_elements; index += 16)
            {
                const int32x4_t q0 = vcvtq_s32_f32(vrndxq_f32(vmulq_n_f32(vld1q_f32(values + index), inverse_scale)));
                const int32x4_t q1 = vcvtq_s32_f32(vrndxq_f32(vmulq_n_f32(vld1q_f32(values + index + 4), inverse_scale)));
                const int32x4_t q2 = vcvtq_s32_f32(vrndxq_f32(vmulq_n_f32(vld1q_f32(values + index + 8), inverse_scale)));
                const int32x4_t q3 = vcvtq_s32_f32(vrndxq_f32(vmulq_n_f32(vld1q_f32(values + index + 12), inverse_scale)));
                const int32x4_t c0 = vmaxq_s32(vminq_s32(q0, maximum_value), minimum_value);
                const int32x4_t c1 = vmaxq_s32(vminq_s32(q1, maximum_value), minimum_value);
                const int32x4_t c2 = vmaxq_s32(vminq_s32(q2, maximum_value), minimum_value);
                const int32x4_t c3 = vmaxq_s32(vminq_s32(q3, maximum_value), minimum_value);
                const int16x8_t c01 = vcombine_s16(vmovn_s32(c0), vmovn_s32(c1));
                const int16x8_t c23 = vcombine_s16(vmovn_s32(c2), vmovn_s32(c3));
                const int8x16_t packed = vcombine_s8(vmovn_s16(c01), vmovn_s16(c23));
                vst1q_s8(quantized + index, packed);
                const int16_t sum = vaddlvq_s8(packed);
                std::memcpy(encoded + 260 + index / 16 * sizeof(sum), &sum, sizeof(sum));
            }
        }
        else
#endif
        {
            for (uint32_t index = 0; index < qnk_block_elements; ++index)
            {
                const long rounded = std::lrintf(inverse_scale * values[index]);
                quantized[index] = static_cast<int8_t>(std::clamp<long>(rounded, -127, 127));
            }
            for (uint32_t sum_index = 0; sum_index < qnk_block_elements / 16; ++sum_index)
            {
                int16_t sum = 0;
                for (uint32_t lane = 0; lane < 16; ++lane)
                    sum = static_cast<int16_t>(sum + quantized[sum_index * 16 + lane]);
                std::memcpy(encoded + 260 + sum_index * sizeof(int16_t), &sum, sizeof(sum));
            }
        }
    }
}

bool qnk_pack_weights(const uint8_t* raw,
                      size_t raw_bytes,
                      DType dtype,
                      size_t rows,
                      uint32_t columns,
                      QnKPack& output) noexcept
{
    const uint64_t expected = qnk_storage_bytes(dtype, rows, columns);
    const uint64_t packed_bytes = qnk_packed_storage_bytes(dtype, rows, columns);
    if (!raw || expected == 0 || expected != raw_bytes || packed_bytes == 0
        || packed_bytes > std::numeric_limits<size_t>::max())
        return false;
    const uint32_t block_count = columns / qnk_block_elements;
    const size_t tile_count = (rows + 7) / 8;
    const size_t block_bytes = qnk_block_bytes(dtype);
    QnKPack packed;
    packed.dtype = dtype;
    packed.rows = rows;
    packed.columns = columns;
    packed.block_count = block_count;
    packed.tile_rows = 8;
    packed.storage.resize(static_cast<size_t>(packed_bytes), 0);
    for (size_t tile = 0; tile < tile_count; ++tile)
    {
        for (uint32_t block = 0; block < block_count; ++block)
        {
            for (uint32_t lane = 0; lane < 8; ++lane)
            {
                const size_t row = tile * 8 + lane;
                uint8_t* destination = packed.storage.data()
                                       + (((tile * block_count + block) * 8 + lane) * block_bytes);
                if (row < rows)
                {
                    const uint8_t* source = raw
                                            + ((row * static_cast<size_t>(block_count) + block) * block_bytes);
                    std::memcpy(destination, source, block_bytes);
                }
            }
        }
    }
    output = std::move(packed);
    return true;
}

bool forward_linear_qnk(const TensorData& matrix,
                        const ActivationBuffer& input,
                        ActivationBuffer& output,
                        bool use_packed_weights,
                        std::shared_ptr<const QnKPack>* sidecar) noexcept
{
    if (!is_qnk_dtype(matrix.dtype) || matrix.shape.size() != 2)
        return false;
    const size_t rows = matrix.shape[0];
    const uint32_t columns = matrix.shape[1];
    if (!qnk_shape_supported(matrix.dtype, rows, columns) || input.columns() != columns)
        return false;
    const std::span<const uint8_t> raw = matrix.qnk_values();
    if (raw.size() != qnk_storage_bytes(matrix.dtype, rows, columns))
        return false;
    output.reset(input.rows(), static_cast<uint32_t>(rows), false);
    if (input.rows() == 0)
        return true;
    if (!use_packed_weights)
    {
        qnk_gemm(raw.data(), nullptr, matrix.dtype, rows, columns,
                 input.row(0), columns, input.rows(), output.row(0), output.columns());
        return true;
    }
    const std::shared_ptr<const QnKPack> packed = get_qnk_packed_weights(matrix, sidecar, rows, columns);
    if (!packed)
        return false;

#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
    {
        msvc_avx512_qnk_gemm(*packed, input.row(0), columns, input.rows(), output.row(0), output.columns());
        return true;
    }
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
    {
        msvc_avx2_qnk_gemm(*packed, input.row(0), columns, input.rows(), output.row(0), output.columns());
        return true;
    }
#endif
    qnk_gemm(nullptr, packed.get(), matrix.dtype, rows, columns,
             input.row(0), columns, input.rows(), output.row(0), output.columns());
    return true;
}

} // namespace moe
} // namespace ncnn
