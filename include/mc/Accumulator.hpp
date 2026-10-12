#pragma once

#include "mc/ObservableBatch.hpp"

#include <algorithm>
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

// Binning: samples go into contiguous blocks of binSize consecutive samples.
// When all numBins blocks are full, neighbouring pairs are merged and binSize
// doubles, so there are always between numBins/2 and numBins complete,
// equal-sized blocks (plus one partially filled block).  Errors use only the
// complete blocks; means use every sample.
struct ObservableData {
    std::vector<double> binSums;
    std::vector<std::size_t> binCounts;
    double totalSum = 0.0;
    std::size_t totalCount = 0;
    std::size_t binSize = 1;

    std::size_t numFullBins() const {
        return binSums.empty() ? 0 : totalCount / binSize;
    }
};

using DerivedFunction =
    std::function<double(const std::unordered_map<std::string, double>&)>;

class ObservableAccumulator {
public:
    explicit ObservableAccumulator(int numBins)
        : numBins_(numBins) {
        if (numBins < 2 || numBins % 2 != 0)
            throw std::runtime_error("numBins must be an even number >= 2");
    }

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

            for (std::size_t i = 0; i < n; ++i)
                addSample(obs, values[i]);

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

    // Means of the complete (equal-sized, contiguous) bins.
    std::vector<double> primaryBinMeans(const std::string& name) const {
        const auto& obs = data_.at(name);

        std::vector<double> out;
        for (std::size_t b = 0; b < obs.numFullBins(); ++b)
            out.push_back(obs.binSums[b] / static_cast<double>(obs.binCounts[b]));
        return out;
    }

    // Leave-one-bin-out values of f over the complete bins.  All observables
    // are expected to be added in every batch, so their bins hold the same
    // samples; the number of jackknife bins is the smallest common count.
    std::vector<double> derivedJackknifeValues(const DerivedFunction& f) const {
        std::vector<double> out;
        if (data_.empty()) return out;

        std::size_t nfull = static_cast<std::size_t>(-1);
        for (const auto& [name, obs] : data_)
            nfull = std::min(nfull, obs.numFullBins());

        std::unordered_map<std::string, double> fullSum, fullCount;
        for (const auto& [name, obs] : data_) {
            double s = 0.0, c = 0.0;
            for (std::size_t b = 0; b < nfull; ++b) {
                s += obs.binSums[b];
                c += static_cast<double>(obs.binCounts[b]);
            }
            fullSum[name] = s;
            fullCount[name] = c;
        }

        for (std::size_t omit = 0; omit < nfull; ++omit) {
            std::unordered_map<std::string, double> means;
            bool valid = true;
            for (const auto& [name, obs] : data_) {
                double count = fullCount[name] - static_cast<double>(obs.binCounts[omit]);
                if (count <= 0.0) { valid = false; break; }
                means[name] = (fullSum[name] - obs.binSums[omit]) / count;
            }
            if (valid) out.push_back(f(means));
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
    void addSample(ObservableData& obs, double v) const {
        std::size_t b = obs.totalCount / obs.binSize;
        if (b == static_cast<std::size_t>(numBins_)) {
            // all bins full: merge neighbouring pairs, double the bin size
            const std::size_t half = static_cast<std::size_t>(numBins_) / 2;
            for (std::size_t k = 0; k < half; ++k) {
                obs.binSums[k]   = obs.binSums[2 * k]   + obs.binSums[2 * k + 1];
                obs.binCounts[k] = obs.binCounts[2 * k] + obs.binCounts[2 * k + 1];
            }
            for (std::size_t k = half; k < static_cast<std::size_t>(numBins_); ++k) {
                obs.binSums[k] = 0.0;
                obs.binCounts[k] = 0;
            }
            obs.binSize *= 2;
            b = half;
        }
        obs.binSums[b] += v;
        obs.binCounts[b] += 1;
        obs.totalSum += v;
        obs.totalCount += 1;
    }

    int numBins_;
    std::size_t totalCount_ = 0;

    std::unordered_map<std::string, ObservableData> data_;
    std::unordered_map<std::string, DerivedFunction> derived_;

    std::set<std::string> rawSampleNames_;
    std::unordered_map<std::string, std::vector<double>> rawSamples_;
};

}