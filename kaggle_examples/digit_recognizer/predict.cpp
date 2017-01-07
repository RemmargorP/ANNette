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

int main() {
  srand(time(NULL));

  ifstream in("net.dump");

  std::stringstream tmp;
  tmp << in.rdbuf();

  ANNette::Network *net = ANNette::Network::load(tmp.str());

  ofstream out("submission.csv");
  out << "ImageId,Label" << endl;

  ifstream test("test.csv");

  for (int i = 1; i <= 28000; ++i) {
    out << i << ',';
    vector<float> data(PIXELS);
    char tmpc;
    for (size_t j = 0; j < PIXELS; ++j) {
      if (j) test >> tmpc;
      test >> data[j];
      data[j] /= 256;
    }

    auto res = net->calculate(data);
    int mx = 0;
    for (int j = 0; j < 10; ++j) {
      if (res[j] > res[mx]) mx = j;
    }

    out << mx << endl;

    if (i % 500 == 0) cerr << "\rDone #" << i;
  }
  cerr << endl;

  return 0;
}
