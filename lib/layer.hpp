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
    Layer(std::vector<Neuron*> d);

    void calculate();
    void dependOn(Layer *target);
    void setValues(const std::vector<float> &v);

    size_t size() const;
    Neuron* getNeuron(size_t i) const;

    std::string dump() const;
  };

}

#endif /* end of include guard: __LAYER_H__ */
