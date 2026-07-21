// Source: https://github.com/NVIDIA/TensorRT/tree/8.5.3/plugin/layerNormPlugin
// Modified by romiptr
#include "trt_layer_norm.hpp"

#include <assert.h>

#include <chrono>
#include <numeric>
#include <functional>

#include "trt_layer_norm_kernel.hpp"
#include "trt_serialize.hpp"

using namespace nvinfer1;

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TRTLayerNormalization"};
}  // namespace

TRTLayerNormalization::TRTLayerNormalization(const std::string &name, 
                                              float epsilon, int axis)
    : TRTPluginBase(name),
      mEpsilon(epsilon),
      mAxis(axis) {}

TRTLayerNormalization::TRTLayerNormalization(const std::string name, 
                                             const void *data, 
                                             size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mEpsilon);
  deserialize_value(&data, &length, &mAxis);
}

nvinfer1::IPluginV2DynamicExt *TRTLayerNormalization::clone() const TRT_NOEXCEPT {
  TRTLayerNormalization *plugin = new TRTLayerNormalization(
      mLayerName, mEpsilon, mAxis);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTLayerNormalization::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  return inputs[0];
}

bool TRTLayerNormalization::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (pos == 0) {
    return ((ioDesc[0].type == nvinfer1::DataType::kFLOAT || 
             ioDesc[0].type == nvinfer1::DataType::kHALF) && 
            (ioDesc[0].format == nvinfer1::TensorFormat::kLINEAR));
  } 
  return (ioDesc[pos].type == ioDesc[0].type) && (ioDesc[pos].format == ioDesc[0].format);
}

void TRTLayerNormalization::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) TRT_NOEXCEPT {}

size_t TRTLayerNormalization::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int TRTLayerNormalization::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                      const nvinfer1::PluginTensorDesc* outputDesc,
                                      const void* const* inputs, void* const* outputs,
                                      void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  const auto inputNbDims = inputDesc[0].dims.nbDims;
  ASSERT(mAxis >= -inputNbDims && mAxis < inputNbDims);
  const auto normAxisNonNegative = mAxis >= 0 ? mAxis : (inputNbDims + mAxis);

  int64_t gridSize = std::accumulate(inputDesc[0].dims.d, inputDesc[0].dims.d + normAxisNonNegative, int64_t{1}, std::multiplies<int64_t>{});
  int64_t nHiddenSize = std::accumulate(inputDesc[0].dims.d + normAxisNonNegative, inputDesc[0].dims.d + inputNbDims, int64_t{1}, std::multiplies<int64_t>{});
  int status = -1;

  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT: {
      const auto input = static_cast<const float*>(inputs[0]);
      const auto gamma = static_cast<const float*>(inputs[1]);
      const auto beta = static_cast<const float*>(inputs[2]);
      auto output = static_cast<float*>(outputs[0]);

      TRTLayerNormalizationKernelLauncher<float>(
          static_cast<int>(gridSize), static_cast<int>(nHiddenSize), 
          input, gamma, beta, output, mEpsilon, stream);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      const auto input = static_cast<const half*>(inputs[0]);
      const auto gamma = static_cast<const half*>(inputs[1]);
      const auto beta = static_cast<const half*>(inputs[2]);
      auto output = static_cast<half*>(outputs[0]);

      TRTLayerNormalizationKernelLauncher<half>(
          static_cast<int>(gridSize), static_cast<int>(nHiddenSize), 
          input, gamma, beta, output, mEpsilon, stream);
      break;
    }
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TRTLayerNormalization::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTLayerNormalization::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTLayerNormalization::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int TRTLayerNormalization::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTLayerNormalization::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mEpsilon) + serialized_size(mAxis);
}

void TRTLayerNormalization::serialize(void* buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mEpsilon);
  serialize_value(&buffer, mAxis);
}

// TRTLayerNormalizationCreator methods
TRTLayerNormalizationCreator::TRTLayerNormalizationCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

TRTLayerNormalizationCreator::~TRTLayerNormalizationCreator() {}

const char* TRTLayerNormalizationCreator::getPluginName() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char* TRTLayerNormalizationCreator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *TRTLayerNormalizationCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  const PluginField* fields = fc->fields;
  float epsilon = 1e-5;
  int axis = -1;

  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "epsilon")) {
      epsilon = *(static_cast<const float*>(fields[i].data));
    }
    if (!strcmp(attrName, "axis")) {
      axis = *(static_cast<const int*>(fields[i].data));
    }
  }

  TRTLayerNormalization *plugin = new TRTLayerNormalization(name, epsilon, axis);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTLayerNormalizationCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new TRTLayerNormalization(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(TRTLayerNormalizationCreator);
}  // namespace mmdeploy