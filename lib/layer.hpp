#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>

#include "neuron.hpp"

namespace ANNette {

  class Layer {
  private:
    std::vector<Neuron*> neurons;
  public:
    Layer(size_t cnt);

    void calculate();
    void dependOn(Layer *target);

    size_t size() const;
    Neuron* getNeuron(size_t i) const;
  };

}

#endif /* end of include guard: __LAYER_H__ */
