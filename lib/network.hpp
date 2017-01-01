#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <vector>

#include "layer.hpp"

namespace ANNette {

  // Input layer is out of Network
  // dep: inp <- network[]
  // network.back() - output layer

  class Network {
  private:
    Layer *input;
    std::vector<Layer*> layers;
  public:
    Network(Layer *input);

    void addLayer(Layer* layer);

    Layer* front() const;
    Layer* back() const;
    Layer* getLayer(size_t cnt) const;

    std::vector<float> calculate(const std::vector<float> &data);
    void updateWeights(float learningRate);
    void backPropagate(const std::vector<float> &expected); // expected.size() == back().size() !!

  };

}

#endif /* end of include guard: __NETWORK_H__ */
