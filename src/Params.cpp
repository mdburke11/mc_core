#include "mc/Params.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace {

std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };

    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());

    return s;
}

}

namespace mc {

Params::Params(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Could not open params file: " + filename);
    }

    std::string line;
    while (std::getline(in, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }

        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("Invalid params line: " + line);
        }

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        values_[key] = val;
    }
}

bool Params::contains(const std::string& key) const {
    return values_.find(key) != values_.end();
}

std::string Params::getString(const std::string& key) const {
    auto it = values_.find(key);
    if (it == values_.end()) {
        throw std::runtime_error("Missing required parameter: " + key);
    }

    markUsed(key);
    return it->second;
}

void Params::markUsed(const std::string& key) const {
    usedKeys_.insert(key);
}

std::vector<std::string> Params::unusedKeys() const {
    std::vector<std::string> out;

    for (const auto& [key, value] : values_) {
        if (usedKeys_.find(key) == usedKeys_.end()) {
            out.push_back(key);
        }
    }

    return out;
}

void Params::validateUnused() const {
    auto unused = unusedKeys();

    if (!unused.empty()) {
        std::string msg = "Unused/unknown parameter(s):\n";
        for (const auto& key : unused) {
            msg += "  - " + key + "\n";
        }

        throw std::runtime_error(msg);
    }
}

int Params::getInt(const std::string& key) const {
    return std::stoi(getString(key));
}

double Params::getDouble(const std::string& key) const {
    return std::stod(getString(key));
}

bool Params::getBool(const std::string& key) const {
    std::string v = getString(key);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);

    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;

    throw std::runtime_error("Invalid bool parameter for key: " + key);
}

std::string Params::getString(const std::string& key, const std::string& fallback) const {
    return contains(key) ? getString(key) : fallback;
}

int Params::getInt(const std::string& key, int fallback) const {
    return contains(key) ? getInt(key) : fallback;
}

double Params::getDouble(const std::string& key, double fallback) const {
    return contains(key) ? getDouble(key) : fallback;
}

bool Params::getBool(const std::string& key, bool fallback) const {
    return contains(key) ? getBool(key) : fallback;
}

RunParams RunParams::from(const Params& p) {
    RunParams r;

    r.numSamples = p.getInt("numSamples", r.numSamples);
    r.thermSweeps = p.getInt("thermSweeps", r.thermSweeps);
    r.sweepsPerBlock = p.getInt("sweepsPerBlock", r.sweepsPerBlock);
    r.measInterval = p.getInt("measInterval", r.measInterval);
    r.checkpointInterval = p.getInt("checkpointInterval", r.checkpointInterval);
    r.numBins = p.getInt("numBins", r.numBins);

    r.resume = p.getBool("resume", r.resume);

    r.outname = p.getString("outname", r.outname);
    r.checkpointFile = p.getString("checkpointFile", r.checkpointFile);

    return r;
}

}