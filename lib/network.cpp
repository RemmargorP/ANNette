#include "network.hpp"

namespace ANNette {

  Network::Network(Layer *input) : input(input) {}

  void Network::addLayer(Layer *layer) {
    layer->dependOn(layers.empty() ? input : layers.back());
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

  std::vector<float> Network::calculate(const std::vector<float> &data) {
    input->setValues(data);

    for (size_t i = 0; i < layers.size(); ++i)
      layers[i]->calculate();

    std::vector<float> ans(back()->size());
    for (size_t i = 0; i < ans.size(); ++i) {
      ans[i] = back()->getNeuron(i)->getValue();
    }

    return ans;
  }

  void Network::updateWeights(float learningRate) {
    for (size_t i = 0; i < layers.size(); ++i) {
      Layer *from = i == 0 ? input : layers[i - 1];
      Layer *cur = layers[i];

      std::vector<float> inp(from->size());
      for (size_t j = 0; j < inp.size(); ++j)
        inp[j] = from->getNeuron(j)->getValue();

      for (size_t j = 0; j < cur->size(); ++j) {
        Neuron *t = cur->getNeuron(j);
        for (size_t k = 0; k < inp.size(); ++k)
          t->changeWeight(from->getNeuron(k), learningRate * t->getDelta() * inp[k]);
        t->changeBias(learningRate * t->getDelta());
      }
    }
  }

  void Network::backPropagate(const std::vector<float> &expected) {
    for (int i = (int)layers.size() - 1; i > -1; i--) {
      Layer *l = layers[i];
      std::vector<float> errors(l->size());

      if (i != (int)layers.size() - 1) {
        for (size_t j = 0; j < l->size(); ++j) {
          float err = 0;
          Neuron *n = l->getNeuron(j);

          for (size_t k = 0; k < layers[i + 1]->size(); ++k) {
            Neuron *tgt = layers[i + 1]->getNeuron(k);
            err += tgt->getWeight(n) * tgt->getDelta();
          }

          errors[j] = err;
        }
      } else {
        for (size_t j = 0; j < l->size(); ++j)
          errors[j] = expected[j] - l->getNeuron(j)->getValue();
      }

      for (size_t j = 0; j < l->size(); ++j) {
        Neuron *n = l->getNeuron(j);
        n->setDelta(errors[j] * n->activateDerivative());
      }
    }
  }

}
