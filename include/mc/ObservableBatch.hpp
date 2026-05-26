#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace mc {

class ObservableBatch {
public:
    void add(const std::string& name, double value) {
        data_[name].push_back(value);
    }

    const std::unordered_map<std::string, std::vector<double>>& data() const {
        return data_;
    }

    std::size_t size() const {
        if (data_.empty()) return 0;
        return data_.begin()->second.size();
    }

    void clear() {
        data_.clear();
    }

private:
    std::unordered_map<std::string, std::vector<double>> data_;
};

}