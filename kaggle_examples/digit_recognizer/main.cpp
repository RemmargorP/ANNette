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

int main() {
  srand(time(NULL));

  ifstream in("net.dump");

  std::stringstream tmp;
  tmp << in.rdbuf();

  ANNette::Network *net = ANNette::Network::load(tmp.str());

  // ANNette::Network *net = new ANNette::Network(new ANNette::Layer(PIXELS));
  // net->addLayer(new ANNette::Layer(150));
  // net->addLayer(new ANNette::Layer(10));

  int sets = 42000;
  Data dataset = readData(sets);

  const size_t epochs = 20;

  for (size_t i = 0; i < epochs; ++i) {
    float err = epoch(net, dataset, 0.05);
    cerr << "\r                                                 \rEpoch #" << i << ' ' << err << endl;

    ofstream out("net.dump");
    out << net->dump() << endl;
    out.close();
  }

  // ofstream out("submission.csv");
  // out << "ImageId,Label" << endl;
  //
  // ifstream test("test.csv");
  //
  // for (int i = 1; i <= 28000; ++i) {
  //   out << i << ',';
  //   vector<float> data(PIXELS);
  //   char tmpc;
  //   for (size_t j = 0; j < PIXELS; ++j) {
  //     if (j) test >> tmpc;
  //     test >> data[j];
  //     data[j] /= 256;
  //   }
  //
  //   auto res = net->calculate(data);
  //   int mx = 0;
  //   for (int j = 0; j < 10; ++j) {
  //     if (res[j] > res[mx]) mx = j;
  //   }
  //
  //   out << mx << endl;
  //
  //   if (i % 500 == 1) cerr << "\rDone #" << i;
  // }
  // cerr << endl;

  return 0;
}
