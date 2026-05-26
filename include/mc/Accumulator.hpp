#pragma once

#include "mc/ObservableBatch.hpp"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace mc {

struct Stats {
    double mean = 0.0;
    double error = 0.0;
    std::size_t count = 0;
};

class ObservableAccumulator {
public:
    explicit ObservableAccumulator(int numBins)
        : numBins_(numBins) {}

    void addBatch(const ObservableBatch& batch) {
        for (const auto& [name, values] : batch.data()) {
            auto& obs = data_[name];

            for (double x : values) {
                std::size_t b = totalCount_ % numBins_;
                if (obs.binSums.empty()) {
                    obs.binSums.assign(numBins_, 0.0);
                    obs.binCounts.assign(numBins_, 0);
                }

                obs.binSums[b] += x;
                obs.binCounts[b] += 1;
                obs.totalSum += x;
                obs.totalCount += 1;
            }
        }

        totalCount_ += batch.size();
    }

    Stats stats(const std::string& name) const {
        const auto& obs = data_.at(name);

        Stats s;
        s.count = obs.totalCount;
        s.mean = obs.totalSum / static_cast<double>(obs.totalCount);

        std::vector<double> binMeans;
        for (int b = 0; b < numBins_; ++b) {
            if (obs.binCounts[b] > 0) {
                binMeans.push_back(obs.binSums[b] / obs.binCounts[b]);
            }
        }

        if (binMeans.size() > 1) {
            double meanBins = 0.0;
            for (double x : binMeans) meanBins += x;
            meanBins /= binMeans.size();

            double var = 0.0;
            for (double x : binMeans) {
                double dx = x - meanBins;
                var += dx * dx;
            }

            var *= static_cast<double>(binMeans.size() - 1) / binMeans.size();
            s.error = std::sqrt(var);
        }

        return s;
    }

    const auto& rawData() const {
        return data_;
    }

private:
    struct ObsData {
        std::vector<double> binSums;
        std::vector<std::size_t> binCounts;
        double totalSum = 0.0;
        std::size_t totalCount = 0;
    };

    int numBins_;
    std::size_t totalCount_ = 0;

    std::unordered_map<std::string, ObsData> data_;
};

}