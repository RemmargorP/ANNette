#include "network.hpp"

namespace ANNette {

  Network::Network() {}

  void Network::addLayer(Layer *layer) {
    layers.push_back(layer);
  }

  Layer* Network::front() const {
    if (!layers.empty()) return layers.front();
    return nullptr;
  }

  Layer* Network::back() const {
    if (!layers.empty()) return layers.back();
    return nullptr;
  }

  Layer* Network::getLayer(size_t cnt) const {
    if (cnt >= layers.size()) return nullptr;
    return layers[cnt];
  }

  void Network::calculate() {
    for (size_t i = 0; i < layers.size(); ++i)
      layers[i]->calculate();
  }

}
