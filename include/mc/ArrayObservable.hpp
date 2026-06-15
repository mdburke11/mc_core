#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace mc {

struct ArrayObservableBatch {
    std::string name;
    std::vector<double> sum;
    std::vector<std::size_t> shape;
    unsigned long long count = 0;

    bool empty() const {
        return count == 0 || sum.empty();
    }
};

struct ArrayAccumulator {
    std::vector<double> sum;
    std::vector<std::size_t> shape;
    unsigned long long count = 0;

    void add(const ArrayObservableBatch& batch) {
        if (batch.empty()) return;

        if (sum.empty()) {
            sum.assign(batch.sum.size(), 0.0);
            shape = batch.shape;
        } else {
            if (sum.size() != batch.sum.size() || shape != batch.shape) {
                throw std::runtime_error(
                    "Array observable shape mismatch for " + batch.name
                );
            }
        }

        for (std::size_t i = 0; i < batch.sum.size(); ++i) {
            sum[i] += batch.sum[i];
        }

        count += batch.count;
    }

    std::vector<double> mean() const {
        if (count == 0) return {};

        std::vector<double> out = sum;
        for (double& x : out) {
            x /= static_cast<double>(count);
        }
        return out;
    }
};

} // namespace mc