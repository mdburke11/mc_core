#pragma once

#include <string>
#include <unordered_map>
#include <set>
#include <vector>

namespace mc {

class Params {
public:
    explicit Params(const std::string& filename);

    bool contains(const std::string& key) const;

    std::string getString(const std::string& key) const;
    int getInt(const std::string& key) const;
    double getDouble(const std::string& key) const;
    bool getBool(const std::string& key) const;

    std::string getString(const std::string& key, const std::string& fallback) const;
    int getInt(const std::string& key, int fallback) const;
    double getDouble(const std::string& key, double fallback) const;
    bool getBool(const std::string& key, bool fallback) const;

    void validateUnused() const;
    std::vector<std::string> unusedKeys() const;

private:
    std::unordered_map<std::string, std::string> values_;
    mutable std::set<std::string> usedKeys_;
    void markUsed(const std::string& key) const;
};

struct RunParams {
    int numSamples = 1000;
    int thermSweeps = 1000;
    int sweepsPerBlock = 10;
    int measInterval = 1;
    int checkpointInterval = 1000;
    int numBins = 100;

    bool resume = false;

    std::string outname = "output.h5";
    std::string checkpointFile = "checkpoint.h5";

    static RunParams from(const Params& p);
};

}