#include "Ising2D.hpp"

#include <cmath>
#include <sstream>

Ising2DParams Ising2DParams::from(const mc::Params& p) {
    Ising2DParams out;
    out.L = p.getInt("L", out.L);
    out.T = p.getDouble("T", out.T);
    out.J = p.getDouble("J", out.J);
    out.seed = static_cast<std::uint64_t>(p.getInt("seed", static_cast<int>(out.seed)));
    out.init = p.getString("init", out.init);
    return out;
}

Ising2D::Ising2D(const Ising2DParams& params)
    : L_(params.L),
      N_(params.L * params.L),
      T_(params.T),
      beta_(1.0 / params.T),
      J_(params.J),
      spins_(N_, 1),
      rng_(params.seed),
      siteDist_(0, N_ - 1) {

    if (params.init == "ordered") {
        std::fill(spins_.begin(), spins_.end(), 1);
    } else {
        for (auto& s : spins_) {
            s = uniform_(rng_) < 0.5 ? -1 : 1;
        }
    }
}

int Ising2D::index(int x, int y) const {
    x = (x + L_) % L_;
    y = (y + L_) % L_;
    return x * L_ + y;
}

int Ising2D::spin(int x, int y) const {
    return spins_[index(x, y)];
}

void Ising2D::sweep() {
    for (int n = 0; n < N_; ++n) {
        int site = siteDist_(rng_);
        int x = site / L_;
        int y = site % L_;

        int s = spins_[site];

        int nn = spin(x + 1, y)
               + spin(x - 1, y)
               + spin(x, y + 1)
               + spin(x, y - 1);

        double dE = 2.0 * J_ * s * nn;

        if (dE <= 0.0 || uniform_(rng_) < std::exp(-beta_ * dE)) {
            spins_[site] = -s;
        }
    }
}

void Ising2D::runThermalization(int sweeps) {
    for (int i = 0; i < sweeps; ++i) {
        sweep();
    }
}

void Ising2D::runBlock(const mc::BlockSpec& block) {
    int measurements = block.sweepsPerBlock / block.measInterval;
    if (measurements < 1) measurements = 1;

    for (int m = 0; m < measurements; ++m) {
        for (int s = 0; s < block.measInterval; ++s) {
            sweep();
        }
        measure();
    }
}

void Ising2D::measure() {
    double E = totalEnergy() / static_cast<double>(N_);
    double M = magnetization() / static_cast<double>(N_);

    pending_.add("E", E);
    pending_.add("E2", E * E);
    pending_.add("M", M);
    pending_.add("absM", std::abs(M));
    pending_.add("M2", M * M);
}

mc::ObservableBatch Ising2D::fetchObservables() {
    mc::ObservableBatch out = pending_;
    pending_.clear();
    return out;
}

double Ising2D::totalEnergy() const {
    double E = 0.0;

    for (int x = 0; x < L_; ++x) {
        for (int y = 0; y < L_; ++y) {
            int s = spin(x, y);
            E -= J_ * s * (spin(x + 1, y) + spin(x, y + 1));
        }
    }

    return E;
}

double Ising2D::magnetization() const {
    double M = 0.0;
    for (int s : spins_) M += s;
    return M;
}

void Ising2D::saveCheckpoint(HighFive::Group& g) const {
    g.createDataSet("spins", spins_);
    g.createDataSet("L", L_);
    g.createDataSet("T", T_);
    g.createDataSet("J", J_);

    std::ostringstream ss;
    ss << rng_;
    g.createDataSet("rng_state", ss.str());
}

void Ising2D::loadCheckpoint(const HighFive::Group& g) {
    g.getDataSet("spins").read(spins_);
    g.getDataSet("T").read(T_);
    beta_ = 1.0 / T_;
    g.getDataSet("J").read(J_);

    std::string rngState;
    g.getDataSet("rng_state").read(rngState);

    std::istringstream ss(rngState);
    ss >> rng_;
}

void Ising2D::writeMetadata(HighFive::Group& g) const {
    g.createAttribute("model", std::string("Ising2D"));
    g.createAttribute("L", L_);
    g.createAttribute("T", T_);
    g.createAttribute("J", J_);
}