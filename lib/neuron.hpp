#ifndef __NEURON_H__
#define __NEURON_H__

#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

namespace ANNette {

  class Neuron {
  protected:
    float out, bias;
    float delta;
    std::unordered_map<Neuron*, float> in;
  public:

    Neuron();
    ~Neuron();

    void dependOn(Neuron *target);
    void deleteDependency(Neuron *target);
    float getWeight(Neuron *target);

    float calculate();
    float activate(float value) const;
    float activateDerivative() const;

    void changeWeight(Neuron *target, float delta);
    void changeBias(float delta);

    float getValue() const;
    void setValue(float v);
    float getDelta() const;
    void setDelta(float v);

    std::string dump() const;
    void load(float bias, std::vector<std::pair<Neuron*, float>> deps);
  };

};

#endif
