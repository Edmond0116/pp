#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>
#include <cstdio>
#include <limits>
using namespace std;

int main(int argc, char **argv) {
    std::random_device rnd;
    std::mt19937 twist(rnd());
    std::uniform_real_distribution<> distr(-100000000., 100000000.);
    int N = atoi(argv[1]);
    float *A = new float[N];
    ofstream testcase, sorted;
    testcase.open("testcase" + string(argv[1]), ios::out | ios::binary);
    for (int i = 0; i < N; ++i) {
        A[i] = distr(twist);
        testcase.write(reinterpret_cast<const char*>(&A[i]), sizeof(float));
    }
    testcase.close();
    sort(A, A + N);
    sorted.open("sorted" + string(argv[1]), ios::out | ios::binary);
    for (int i = 0; i < N; ++i)
        sorted.write(reinterpret_cast<const char*>(&A[i]), sizeof(float));
    sorted.close();
    delete[] A;
}
