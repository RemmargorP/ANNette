#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "lib/annette.hpp"

using namespace std;

int main() {
  srand(time(NULL));

  float mn = 1;

  for (int i = 0; i < 10000; ++i) {
    ANNette::Neuron n;
    mn = min(mn, n.calculate());
  }

  cout << mn << endl;
  return 0;
}
