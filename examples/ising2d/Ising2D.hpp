#pragma once

#include "mc/Params.hpp"
#include "mc/ObservableBatch.hpp"
#include "mc/Runner.hpp"
#include "mc/H5IO.hpp"

#include <highfive/H5Group.hpp>

#include <random>
#include <vector>
#include <cstdint>

struct Ising2DParams {
    int L = 16;
    double T = 2.269185;
    double J = 1.0;
    std::uint64_t seed = 1234;

    static Ising2DParams from(const mc::Params& p);
};

class Ising2D {
public:
    explicit Ising2D(const Ising2DParams& params);

    void runThermalization(int sweeps);
    void runBlock(const mc::BlockSpec& block);
    mc::ObservableBatch fetchObservables();

    void saveCheckpoint(mc::H5Writer& writer) const;
    void loadCheckpoint(mc::H5Reader& reader);

    void writeMetadata(HighFive::Group& g) const;

private:
    int index(int x, int y) const;
    int spin(int x, int y) const;

    void sweep();
    void measure();
    double totalEnergy() const;
    double magnetization() const;

    int L_;
    int N_;
    double T_;
    double beta_;
    double J_;

    std::vector<int> spins_;

    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
    std::uniform_int_distribution<int> siteDist_;

    mc::ObservableBatch pending_;
};