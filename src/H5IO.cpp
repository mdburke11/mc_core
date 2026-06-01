#include "mc/H5IO.hpp"

#include <stdexcept>

namespace mc {

H5Writer::H5Writer(const std::string& filename)
    : file_(filename, HighFive::File::Overwrite) {}

H5Reader::H5Reader(const std::string& filename)
    : file_(filename, HighFive::File::ReadOnly) {}

HighFive::Group H5Writer::getOrCreateGroup(
    const std::string& path
) {
    if (path.empty() || path == "/") {
        return file_.getGroup("/");
    }

    std::string current;

    for (std::size_t i = 1; i < path.size(); ++i) {

        if (path[i] == '/') {

            if (!file_.exist(current)) {
                file_.createGroup(current);
            }
        }

        current += path[i];
    }

    if (!file_.exist(path)) {
        file_.createGroup(path);
    }

    return file_.getGroup(path);
}

void H5Writer::writeRunParams(const RunParams& p) {
    auto g = getOrCreateGroup("/metadata/run");

    g.createAttribute("numSamples", p.numSamples);
    g.createAttribute("thermSweeps", p.thermSweeps);
    g.createAttribute("sweepsPerBlock", p.sweepsPerBlock);
    g.createAttribute("measInterval", p.measInterval);
    g.createAttribute("checkpointInterval", p.checkpointInterval);
    g.createAttribute("numBins", p.numBins);
    g.createAttribute("resume", static_cast<int>(p.resume));
    g.createAttribute("outname", p.outname);
    g.createAttribute("checkpointFile", p.checkpointFile);
}

void H5Writer::writeAccumulator(const ObservableAccumulator& acc) {
    auto g = getOrCreateGroup("/observables");

    for (const auto& [name, data] : acc.rawData()) {
        auto stats = acc.stats(name);

        auto obsGroup = g.createGroup(name);
        obsGroup.createAttribute("mean", stats.mean);
        obsGroup.createAttribute("error", stats.error);
        obsGroup.createAttribute("count", static_cast<unsigned long long>(stats.count));

        obsGroup.createDataSet("bin_sums", data.binSums);
        obsGroup.createDataSet("bin_counts", data.binCounts);
    }

    for (const auto& [name, f] : acc.derivedObservables()) {
        auto stats = acc.derivedStats(name);

        auto obsGroup = g.createGroup(name);
        obsGroup.createAttribute("mean", stats.mean);
        obsGroup.createAttribute("error", stats.error);
        obsGroup.createAttribute("count", static_cast<unsigned long long>(stats.count));
        obsGroup.createAttribute("derived", 1);
    }
}

void H5Writer::writeAccumulatorCheckpoint(
        const ObservableAccumulator& acc,
        const std::string& basePath
    ) {
        auto g = getOrCreateGroup(basePath);

        writeScalar(
            basePath + "/total_count",
            static_cast<unsigned long long>(acc.totalCount())
        );

        for (const auto& [name, data] : acc.rawData()) {
            auto obsPath = basePath + "/" + name;
            auto obsGroup = getOrCreateGroup(obsPath);

            obsGroup.createDataSet("bin_sums", data.binSums);
            obsGroup.createDataSet("bin_counts", data.binCounts);
            obsGroup.createAttribute("total_sum", data.totalSum);
            obsGroup.createAttribute(
                "total_count",
                static_cast<unsigned long long>(data.totalCount)
            );
        }
    }

int H5Reader::readInt(const std::string& path) const {
    int value;
    file_.getDataSet(path).read(value);
    return value;
}

}