#pragma once

#include "mc/Params.hpp"
#include "mc/ObservableBatch.hpp"
#include "mc/Accumulator.hpp"
#include "mc/H5IO.hpp"

#include <iostream>

namespace mc {

struct BlockSpec {
    int sweepsPerBlock = 1;
    int measInterval = 1;
};

template <class ModelT>
class Runner {
public:
    Runner(ModelT& model, const RunParams& params)
        : model_(model),
          params_(params),
          accumulator_(params.numBins) {}

    void run() {
        if (params_.resume) {
            loadCheckpoint();
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

            if (completedSamples_ % 1000 == 0) {
                std::cout << "Completed samples: "
                          << completedSamples_ << " / "
                          << params_.numSamples << "\n";
            }
        }

        saveFinalOutput();
    }

private:
    void saveFinalOutput() const {
        H5Writer writer(params_.outname);
        writer.writeRunParams(params_);
        writer.writeModelMetadata(model_);
        writer.writeAccumulator(accumulator_);
    }

    void saveCheckpoint() const {
        H5Writer writer(params_.checkpointFile);
        writer.writeScalar("/checkpoint/completed_samples", completedSamples_);
        model_.saveCheckpoint(writer);
        writer.writeAccumulatorCheckpoint(accumulator_);
    }

    void loadCheckpoint() {
        H5Reader reader(params_.checkpointFile);
        completedSamples_ = reader.readInt("/checkpoint/completed_samples");
        model_.loadCheckpoint(reader);
        // accumulator reload can come later
    }

    ModelT& model_;
    RunParams params_;
    ObservableAccumulator accumulator_;

    int completedSamples_ = 0;
};

}