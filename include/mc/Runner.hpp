#pragma once

#include "mc/Params.hpp"
#include "mc/ObservableBatch.hpp"
#include "mc/Accumulator.hpp"
#include "mc/H5IO.hpp"

#include <iostream>
#include <utility>
#include <atomic>
#include <csignal>

namespace mc {

struct BlockSpec {
    int sweepsPerBlock = 1;
    int measInterval = 1;
};

inline std::atomic<bool> stopRequested = false;

inline void signalHandler(int) {
    stopRequested.store(true);
}

template <class ModelT>
class Runner {
public:
    Runner(ModelT& model, const RunParams& params)
        : model_(model),
          params_(params),
          accumulator_(params.numBins) {}

    void run() {

        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);

        if (params_.resume) {
            loadCheckpoint();

            std::cout << "Resumed from checkpoint at sample "
                    << completedSamples_ << " / "
                    << params_.numSamples << "\n";

            if (completedSamples_ >= params_.numSamples) {
                std::cout << "Checkpoint already satisfies requested numSamples. "
                        << "Writing final output and exiting.\n";
            }
        } else {
            std::cout << "Thermalizing for " << params_.thermSweeps << " sweeps\n";
            model_.runThermalization(params_.thermSweeps);
        }

        BlockSpec block;
        block.sweepsPerBlock = params_.sweepsPerBlock;
        block.measInterval = params_.measInterval;

        while (completedSamples_ < params_.numSamples) {
            model_.runBlock(block);

            ObservableBatch batch = model_.fetchObservables();
            accumulator_.addBatch(batch);

            completedSamples_ += static_cast<int>(batch.size());

            if (completedSamples_ % params_.checkpointInterval == 0) {
                saveCheckpoint();
            }

            if (stopRequested.load()) {
                std::cout << "Stop signal received. Saving checkpoint and exiting.\n";
                saveCheckpoint();
                return;
            }

            if (completedSamples_ % 1000 == 0) {
                std::cout << "Completed samples: "
                          << completedSamples_ << " / "
                          << params_.numSamples << "\n";
            }
        }

        saveFinalOutput();
    }

    void addDerivedObservable(const std::string& name, DerivedFunction f) {
        accumulator_.addDerivedObservable(name, std::move(f));
    }

private:
    void saveFinalOutput() const {
        H5Writer writer(params_.outname);
        writer.writeRunParams(params_);
        writer.writeModelMetadata(model_);
        writer.writeAccumulator(accumulator_);
    }

    void saveCheckpoint() const {
        std::cout << "Saving checkpoint at sample "
          << completedSamples_ << "\n";
          
        H5Writer writer(params_.checkpointFile);
        writer.writeScalar("/checkpoint/completed_samples", completedSamples_);
        model_.saveCheckpoint(writer);
        writer.writeAccumulatorCheckpoint(accumulator_);
    }

    void loadCheckpoint() {
        H5Reader reader(params_.checkpointFile);
        completedSamples_ = reader.readInt("/checkpoint/completed_samples");
        model_.loadCheckpoint(reader);

        loadAccumulatorCheckpoint(reader);
    }

    void loadAccumulatorCheckpoint(H5Reader& reader) {
        if (!reader.file().exist("/checkpoint/accumulators")) {
            return;
        }

        auto total =
            reader.readScalar<unsigned long long>("/checkpoint/accumulators/total_count");
        accumulator_.setTotalCount(static_cast<std::size_t>(total));

        auto g = reader.file().getGroup("/checkpoint/accumulators");

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

            accumulator_.rawDataMutable()[name] = std::move(data);
        }
    }

    ModelT& model_;
    RunParams params_;
    ObservableAccumulator accumulator_;

    int completedSamples_ = 0;
};

}