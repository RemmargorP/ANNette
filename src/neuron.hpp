#ifndef __NEURON_H__
#define __NEURON_H__

#include <unordered_set>

namespace Annette {

  class Neuron {
  protected:
    float value;
    unordered_set<pair<Synapse*, float>> in;
    unordered_set<Synapse*> out;
  public:
    Neuron();
    void connectTo(Neuron *to);
    void deleteConnection(int id);
    float calculate();
  };

};

#endif
