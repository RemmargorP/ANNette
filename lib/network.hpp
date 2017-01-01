#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <vector>

#include "layer.hpp"

namespace ANNette {

  class Network {
  private:
    std::vector<Layer*> layers;
  public:
    Network();

    void addLayer(Layer* layer);

    Layer* front() const;
    Layer* back() const;
    Layer* getLayer(size_t cnt) const;

    void calculate();
  };

}

#endif /* end of include guard: __NETWORK_H__ */
