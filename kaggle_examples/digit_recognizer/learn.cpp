#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>

#include "lib/annette.hpp"

using namespace std;

typedef vector<pair<vector<float>, vector<float>>> Data;

const size_t PIXELS = 784;

float epoch(ANNette::Network *net, const Data &dataset, const float lrate) {
  float err = 0;
  for (size_t i = 0; i < dataset.size(); ++i) {
    auto res = net->calculate(dataset[i].first);
    for (size_t j = 0; j < res.size(); ++j)
      err += (dataset[i].second[j] - res[j]) * (dataset[i].second[j] - res[j]);
    net->backPropagate(dataset[i].second);
    net->updateWeights(lrate);

    if (i % 200 == 0) {
      cerr << "\rLearnt #" << i << "th dataset";
    }
  }
  return err;
}

Data readData(size_t sets) {
  Data dataset(sets);

  ifstream inp("train.csv");
  char tmpc;

  for (size_t i = 0; i < sets; ++i) {
    dataset[i].first.resize(PIXELS);
    dataset[i].second.resize(10);

    int x;
    inp >> x;

    dataset[i].second[x] = 1;

    for (size_t j = 0; j < PIXELS; ++j) {
      inp >> tmpc >> dataset[i].first[j];
      dataset[i].first[j] /= 256;
    }

    if (i % 1000 == 0) cerr << "\rRead #" << i;
  }

  cerr << "\rDone reading" << endl;

  inp.close();
  return dataset;
}

int main(int argc, char** argv) {
  srand(time(NULL));

  ANNette::Network *net;

  if (argc > 1) {
    net = new ANNette::Network(new ANNette::Layer(PIXELS));
    net->addLayer(new ANNette::Layer(200));
    net->addLayer(new ANNette::Layer(40));
    net->addLayer(new ANNette::Layer(10));
  } else {
    ifstream in("net.dump");

    std::stringstream tmp;
    tmp << in.rdbuf();

    net = ANNette::Network::load(tmp.str());
  }
  
  int sets = 42000;
  Data dataset = readData(sets);

  for (size_t i = 0; true; ++i) {
    float err = epoch(net, dataset, 0.4);
    cerr << "\r                                                 \rEpoch #" << i << ' ' << err << endl;

    ofstream out("net.dump");
    out << net->dump() << endl;
    out.close();
  }

  return 0;
}
