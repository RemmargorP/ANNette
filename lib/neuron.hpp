#ifndef __NEURON_H__
#define __NEURON_H__

#include <unordered_map>

namespace ANNette {

  class Neuron {
  protected:
    float value, out, bias;
    std::unordered_map<Neuron*, float> in;
  public:
    Neuron();
    ~Neuron();

    void dependOn(Neuron *target);
    void deleteDependency(Neuron *target);

    float calculate();
    float activate(float value) const;
    float getValue() const;
  };

};

#endif
