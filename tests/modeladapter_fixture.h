#ifndef NCNN_MOE_MODELADAPTER_FIXTURE_H
#define NCNN_MOE_MODELADAPTER_FIXTURE_H

#include "ncnn/moe/modeladapter.h"

#include <vector>

namespace ncnn {
namespace moe {

class ActivationBuffer;

[[nodiscard]] std::vector<std::vector<float>> batch_to_vectors(const ActivationBuffer& batch);

class FixtureModelAdapter final : public ModelAdapter
{
public:
    [[nodiscard]] bool can_load(const ModelManifest& manifest) const override;
    [[nodiscard]] Result<MoeModelDescriptor> parse_model(const ModelPackage& package) const override;
    [[nodiscard]] Result<WeightMapping> map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const override;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELADAPTER_FIXTURE_H
