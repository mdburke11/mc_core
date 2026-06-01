#pragma once

#include "mc/Params.hpp"
#include "mc/ObservableBatch.hpp"
#include "mc/Accumulator.hpp"
#include "mc/H5IO.hpp"
#include "mc/AccumulatorIO.hpp"
#include "mc/ModelHooks.hpp"

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
        writer.writeScalar("/checkpoint/runner/completed_samples", completedSamples_);

        auto modelGroup =
            writer.file().createGroup("/checkpoint/replicas/0/model");

        model_.saveCheckpoint(modelGroup);

        writer.writeAccumulatorCheckpoint(accumulator_);
    }

    void loadCheckpoint() {
        H5Reader reader(params_.checkpointFile);

        completedSamples_ =
            reader.readInt("/checkpoint/runner/completed_samples");

        auto modelGroup =
            reader.file().getGroup("/checkpoint/replicas/0/model");

        model_.loadCheckpoint(modelGroup);

        loadAccumulatorCheckpoint(reader, accumulator_);
    }

    ModelT& model_;
    RunParams params_;
    ObservableAccumulator accumulator_;

    int completedSamples_ = 0;
};

} // namespace mc