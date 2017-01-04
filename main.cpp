#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>

#include "lib/annette.hpp"

using namespace std;

void learn(ANNette::Network *net) {
  vector<pair<vector<float>, vector<float>>> dataset;
  dataset.push_back({ {0, 0}, {0} });
  dataset.push_back({ {0, 1}, {1} }); // XOR
  dataset.push_back({ {1, 0}, {1} });
  dataset.push_back({ {1, 1}, {0} });

  int epochs = 100000;
  float lrate = 0.5;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    float err = 0;
    for (size_t i = 0; i < dataset.size(); ++i) {
      auto res = net->calculate(dataset[i].first);
      for (size_t j = 0; j < res.size(); ++j)
        err += (dataset[i].second[j] - res[j]) * (dataset[i].second[j] - res[j]);
      net->backPropagate(dataset[i].second);
      net->updateWeights(lrate);
    }

    if (epoch % 10000 == 0) printf("epoch #%d: lrate: %.3f error: %.5f\n", epoch, lrate, err);
  }

  for (size_t i = 0; i < dataset.size(); ++i) {
    vector<float> res = net->calculate(dataset[i].first);
    printf("TEST data: %ld ^ %ld = %f\n", lround(dataset[i].first[0]), lround(dataset[i].first[1]), res[0]);
  }
}

int main() {
  srand(time(NULL));

  ANNette::Network *net = new ANNette::Network(new ANNette::Layer(2));
  net->addLayer(new ANNette::Layer(3));
  net->addLayer(new ANNette::Layer(1));

  learn(net);

  ofstream out("net.dump");
  out << net->dump() << endl;
  out.close();

  ifstream in("net.dump");

  std::stringstream tmp;
  tmp << in.rdbuf();

  ANNette::Network *sec = ANNette::Network::load(tmp.str());
  in.close();

  learn(sec);

  ofstream outsec("net2.dump");
  outsec << sec->dump() << endl;
  outsec.close();

  return 0;
}
