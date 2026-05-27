#pragma once

#include "mc/Accumulator.hpp"
#include "mc/H5IO.hpp"

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
        if (name == "total_count") continue;

        auto obsGroup = g.getGroup(name);

        ObservableData data;
        obsGroup.getDataSet("bin_sums").read(data.binSums);
        obsGroup.getDataSet("bin_counts").read(data.binCounts);

        unsigned long long count;
        obsGroup.getAttribute("total_count").read(count);
        obsGroup.getAttribute("total_sum").read(data.totalSum);

        data.totalCount = static_cast<std::size_t>(count);

        acc.rawDataMutable()[name] = std::move(data);
    }
}

}