#include <unordered_map>
#include <iostream>

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

  std::string Network::dump() const {
    std::stringstream res;
    res << "network " << (size_t)this << ' ' << layers.size() + 1;
    res << std::endl << input->dump();
    for (auto l : layers)
      res << std::endl << l->dump();
    return res.str();
  }

  Network* Network::load(std::string s) {
    std::stringstream inp;
    inp.str(s);

    std::unordered_map<size_t, Neuron*> neuronAddrConv;
    std::unordered_map<size_t, std::vector<std::pair<size_t, float>>> neuronStructure;
    std::unordered_map<size_t, float> biases;

    std::vector<std::vector<size_t>> structure;

    size_t layers;
    std::string tmp;
    inp >> tmp >> tmp >> layers; // network addr cnt
    // std::cerr << "Layers " << layers << std::endl;

    structure.resize(layers);

    for (size_t i = 0; i < layers; ++i) {
      //layer addr cnt
      size_t neurons;
      inp >> tmp >> tmp >> neurons;

      // std::cerr << "Layer #" << i << ": " << neurons << " neurons" << std::endl;

      structure[i].resize(neurons);
      for (size_t j = 0; j < neurons; ++j) {
        //neuron addr bias cnt cnt*[addr weight]
        size_t addr;
        float bias;
        size_t conns;

        inp >> tmp >> addr >> bias >> conns;

        // std::cerr << "Neuron #" << j << ": " << addr << ' ' << bias << ' ' << conns << std::endl;

        Neuron *tmp = new Neuron();

        biases[addr] = bias;
        neuronAddrConv[addr] = tmp;
        structure[i][j] = addr;

        std::vector<std::pair<size_t, float>> deps(conns);
        for (size_t k = 0; k < conns; ++k) {
          size_t depAddr;
          float w;
          inp >> depAddr >> w;
          deps[k] = {depAddr, w};
        }

        neuronStructure[addr] = deps;

      }
    }

    Network* net = nullptr;

    for (size_t i = 0; i < layers; ++i) {
      std::vector<Neuron*> cur(structure[i].size());

      for (size_t j = 0; j < structure[i].size(); ++j) {

        std::vector<std::pair<Neuron*, float>> deps(neuronStructure[structure[i][j]].size());

        //neuronStructure[structure[i][j]]

        for (size_t k = 0; k < deps.size(); ++k) {
          deps[k].first = neuronAddrConv[neuronStructure[structure[i][j]][k].first];
          deps[k].second = neuronStructure[structure[i][j]][k].second;
        }

        Neuron* n = neuronAddrConv[structure[i][j]];
        n->load(biases[structure[i][j]], deps);

        cur[j] = n;

      }

      if (i == 0) //input
        net = new Network(new Layer(cur));
      else
        net->layers.push_back(new Layer(cur));

    }

    return net;
  }

}
