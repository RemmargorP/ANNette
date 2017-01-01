#include "layer.hpp"

namespace ANNette {

  Layer::Layer(int cnt) {
    neurons.resize(cnt);
    for (int i = 0; i < cnt; ++i) {
      neurons[i] = new Neuron();
    }
  }

  void Layer::calculate() {
    for (size_t i = 0; i < neurons.size(); ++i) {
      neurons[i]->calculate();
    }
  }

  void Layer::dependOn(Layer *target) {
    if (target == nullptr) return;

    for (size_t i = 0; i < neurons.size(); ++i) {
      for (size_t j = 0; j < target->size(); ++j) {
        neurons[i]->dependOn(target->getNeuron(j));
      }
    }
  }

  size_t Layer::size() const {
    return neurons.size();
  }

  Neuron* Layer::getNeuron(size_t i) const {
    if (i >= neurons.size()) return nullptr;
    return neurons[i];
  }

}
