#pragma once

#include "mc/Accumulator.hpp"
#include "mc/H5IO.hpp"

#include <iostream>
#include <string>

namespace mc {

inline void loadAccumulatorCheckpoint(
    H5Reader& reader,
    ObservableAccumulator& acc,
    const std::string& basePath = "/checkpoint/accumulators"
) {
    if (!reader.file().exist(basePath)) {
        return;
    }

    auto total =
        reader.readScalar<unsigned long long>(
            basePath + "/total_count"
        );

    acc.setTotalCount(static_cast<std::size_t>(total));

    auto g = reader.file().getGroup(basePath);

    for (const auto& name : g.listObjectNames()) {
        if (name == "total_count" || name == "raw") continue;

        auto obsGroup = g.getGroup(name);

        ObservableData data;
        obsGroup.getDataSet("bin_sums").read(data.binSums);
        obsGroup.getDataSet("bin_counts").read(data.binCounts);

        unsigned long long count;
        obsGroup.getAttribute("total_count").read(count);
        obsGroup.getAttribute("total_sum").read(data.totalSum);

        data.totalCount = static_cast<std::size_t>(count);

        if (obsGroup.hasAttribute("bin_size")) {
            unsigned long long bs;
            obsGroup.getAttribute("bin_size").read(bs);
            data.binSize = static_cast<std::size_t>(bs);
        } else {
            // Checkpoint from the old interleaved binning (sample i -> bin
            // i % numBins).  Its bins cannot be turned into contiguous blocks;
            // keep them with a bin size consistent with the sample count so
            // that accumulation continues correctly.  Errors for this run
            // still carry the old interleaved-bin caveat.
            const std::size_t nb = data.binSums.size();
            std::size_t bs = 1;
            while (data.totalCount >= nb * bs) bs *= 2;
            data.binSize = bs;
            static bool warned = false;
            if (!warned) {
                std::cerr << "Warning: checkpoint uses interleaved bins "
                             "(written before mc_core switched to contiguous "
                             "blocks); error bars of this run are only "
                             "approximate\n";
                warned = true;
            }
        }

        acc.rawDataMutable()[name] = std::move(data);
    }

    if (acc.hasRawSamples() && reader.file().exist(basePath + "/raw")) {
        auto rawGroup = reader.file().getGroup(basePath + "/raw");
        for (const auto& name : rawGroup.listObjectNames()) {
            if (acc.rawSampleNames().count(name)) {
                std::vector<double> samples;
                rawGroup.getDataSet(name).read(samples);
                acc.rawSamplesMutable()[name] = std::move(samples);
            }
        }
    }
}

}