#pragma once

#include "mc/ObservableBatch.hpp"

#include <cmath>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <stdexcept>

namespace mc {

struct Stats {
    double mean = 0.0;
    double error = 0.0;
    std::size_t count = 0;
};

struct ObservableData {
    std::vector<double> binSums;
    std::vector<std::size_t> binCounts;
    double totalSum = 0.0;
    std::size_t totalCount = 0;
};

using DerivedFunction =
    std::function<double(const std::unordered_map<std::string, double>&)>;

class ObservableAccumulator {
public:
    explicit ObservableAccumulator(int numBins)
        : numBins_(numBins) {}

    void enableRawSamples(const std::set<std::string>& names) {
        rawSampleNames_ = names;
    }

    void addDerivedObservable(const std::string& name, DerivedFunction f) {
        derived_[name] = std::move(f);
    }

    void addBatch(const ObservableBatch& batch) {
        const std::size_t n = batch.size();

        for (const auto& [name, values] : batch.data()) {
            if (values.size() != n) {
                throw std::runtime_error("ObservableBatch has inconsistent observable lengths.");
            }

            auto& obs = data_[name];

            if (obs.binSums.empty()) {
                obs.binSums.assign(numBins_, 0.0);
                obs.binCounts.assign(numBins_, 0);
            }

            for (std::size_t i = 0; i < n; ++i) {
                std::size_t b = (totalCount_ + i) % numBins_;

                obs.binSums[b] += values[i];
                obs.binCounts[b] += 1;
                obs.totalSum += values[i];
                obs.totalCount += 1;
            }

            if (rawSampleNames_.count(name)) {
                auto& raw = rawSamples_[name];
                raw.insert(raw.end(), values.begin(), values.end());
            }
        }

        totalCount_ += n;
    }

    Stats stats(const std::string& name) const {
        const auto& obs = data_.at(name);

        Stats s;
        s.count = obs.totalCount;
        if (obs.totalCount == 0) return s;

        s.mean = obs.totalSum / static_cast<double>(obs.totalCount);

        auto vals = primaryBinMeans(name);
        s.error = binMeanError(vals);

        return s;
    }

    Stats derivedStats(const std::string& name) const {
        auto it = derived_.find(name);
        if (it == derived_.end()) {
            throw std::runtime_error(
                "Unknown derived observable: " + name
            );
        }

        const auto& f = it->second;

        Stats s;

        // ---------------------------------
        // Full-data mean (what we report)
        // ---------------------------------

        std::unordered_map<std::string,double> fullMeans;

        for (const auto& [obsName, obs] : data_) {

            if (obs.totalCount == 0) {
                return s;
            }

            fullMeans[obsName] =
                obs.totalSum /
                static_cast<double>(obs.totalCount);
        }

        s.mean = f(fullMeans);

        // Report measurement count
        if (!data_.empty()) {
            s.count =
                data_.begin()->second.totalCount;
        }

        // ---------------------------------
        // Jackknife error only
        // ---------------------------------

        const auto jackVals =
            derivedJackknifeValues(f);

        s.error =
            jackknifeError(jackVals);

        return s;
    }

    std::vector<double> primaryBinMeans(const std::string& name) const {
        const auto& obs = data_.at(name);

        std::vector<double> out;
        for (int b = 0; b < numBins_; ++b) {
            if (obs.binCounts[b] > 0) {
                out.push_back(obs.binSums[b] / obs.binCounts[b]);
            }
        }
        return out;
    }

    std::vector<double> derivedJackknifeValues(const DerivedFunction& f) const {
        std::vector<double> out;

        for (int omit = 0; omit < numBins_; ++omit) {
            std::unordered_map<std::string, double> means;

            bool valid = true;

            for (const auto& [name, obs] : data_) {
                double sum = obs.totalSum;
                std::size_t count = obs.totalCount;

                if (omit < static_cast<int>(obs.binSums.size())) {
                    sum -= obs.binSums[omit];
                    count -= obs.binCounts[omit];
                }

                if (count == 0) {
                    valid = false;
                    break;
                }

                means[name] = sum / static_cast<double>(count);
            }

            if (valid) {
                out.push_back(f(means));
            }
        }

        return out;
    }

    static double jackknifeError(const std::vector<double>& jackVals) {
        const std::size_t n = jackVals.size();
        if (n <= 1) return 0.0;

        double mean = 0.0;
        for (double x : jackVals) mean += x;
        mean /= static_cast<double>(n);

        double var = 0.0;
        for (double x : jackVals) {
            double dx = x - mean;
            var += dx * dx;
        }

        return std::sqrt((n - 1.0) / n * var);
    }

    static double binMeanError(const std::vector<double>& binMeans) {
        const std::size_t n = binMeans.size();
        if (n <= 1) return 0.0;

        double mean = 0.0;
        for (double x : binMeans) mean += x;
        mean /= static_cast<double>(n);

        double var = 0.0;
        for (double x : binMeans) {
            double dx = x - mean;
            var += dx * dx;
        }

        return std::sqrt(var / (n * (n - 1.0)));
    }

    const auto& rawData() const {
        return data_;
    }

    auto& rawDataMutable() {
        return data_;
    }

    const auto& derivedObservables() const {
        return derived_;
    }

    const auto& rawSamples() const {
        return rawSamples_;
    }

    auto& rawSamplesMutable() {
        return rawSamples_;
    }

    const auto& rawSampleNames() const {
        return rawSampleNames_;
    }

    bool hasRawSamples() const {
        return !rawSampleNames_.empty();
    }

    int numBins() const {
        return numBins_;
    }

    std::size_t totalCount() const {
        return totalCount_;
    }

    void setTotalCount(std::size_t n) {
        totalCount_ = n;
    }

private:
    int numBins_;
    std::size_t totalCount_ = 0;

    std::unordered_map<std::string, ObservableData> data_;
    std::unordered_map<std::string, DerivedFunction> derived_;

    std::set<std::string> rawSampleNames_;
    std::unordered_map<std::string, std::vector<double>> rawSamples_;
};

}