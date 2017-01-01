#ifndef __SYNAPSE_H__
#define __SYNAPSE_H__

#include "neuron.hpp"

namespace Annette {

  class Synapse {
  private:
    Neuron *from, *to;
    float value;

    int id;
  public:
    Synapse(Neuron *from, Neuron *to);

    float read() const;
    void set(float value);

    int getId() const;
  };

  bool operator<(const Synapse *a, const Synapse *b);

};

#endif
