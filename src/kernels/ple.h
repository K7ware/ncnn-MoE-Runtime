#ifndef NCNN_MOE_PLE_H
#define NCNN_MOE_PLE_H

#include "activationbuffer.h"

#include "ncnn/moe/result.h"

#include <cstdint>
#include <span>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
class WeightStore;
struct PleBlockPlan;
struct LayerCache;
struct AttentionScratch;
struct MoeModelDescriptor;

// Hidden must not alias the attention scratch buffers.
[[nodiscard]] Result<void> forward_ple(const WeightStore& weights,
                                       const CompiledOperatorTable& operators,
                                       const PleBlockPlan& plan,
                                       const MoeModelDescriptor& descriptor,
                                       std::span<const int32_t> input_ids,
                                       LayerCache& cache,
                                       AttentionScratch& scratch,
                                       ActivationBuffer& hidden,
                                       uint64_t flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_PLE_H
