#pragma once

#include "mc/Params.hpp"
#include "mc/ObservableBatch.hpp"
#include "mc/Accumulator.hpp"
#include "mc/H5IO.hpp"
#include "mc/AccumulatorIO.hpp"
#include "mc/ModelHooks.hpp"
#include "mc/ArrayObservable.hpp"

#include <iostream>
#include <utility>
#include <atomic>
#include <csignal>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <functional>
#include <highfive/H5File.hpp>

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

            for (const auto& arrBatch : model_.fetchArrayObservables()) {
                arrayAccumulators_[arrBatch.name].add(arrBatch);
            }

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

    void writeArrayAccumulatorCheckpoint(H5Writer& writer) const {
        for (const auto& [name, acc] : arrayAccumulators_) {
            const std::string base =
                "/checkpoint/array_observables/" + name;

            writer.writeScalar(base + "/count", acc.count);

            if (acc.count > 0 && !acc.sum.empty()) {
                writer.writeArray(base + "/sum", acc.sum, acc.shape);

                std::vector<unsigned long long> shape(
                    acc.shape.begin(),
                    acc.shape.end()
                );

                writer.writeVector(base + "/shape", shape);
            }
        }
    }

    void loadArrayAccumulatorCheckpoint(H5Reader& reader) {
        const std::string root = "/checkpoint/array_observables";

        if (!reader.file().exist(root)) {
            return;
        }

        std::function<void(const std::string&, const std::string&)> visit;

        visit = [&](const std::string& h5Path, const std::string& logicalName) {
            auto g = reader.file().getGroup(h5Path);

            if (g.exist("count")) {
                ArrayAccumulator acc;

                acc.count =
                    reader.readScalar<unsigned long long>(h5Path + "/count");

                if (
                    acc.count > 0 &&
                    reader.file().exist(h5Path + "/sum")
                ) {
                    reader.file().getDataSet(h5Path + "/sum").read(acc.sum);

                    std::vector<unsigned long long> shape;
                    reader.file().getDataSet(h5Path + "/shape").read(shape);

                    acc.shape.assign(shape.begin(), shape.end());
                }

                arrayAccumulators_[logicalName] = std::move(acc);
                return;
            }

            for (const auto& child : g.listObjectNames()) {
                const std::string childH5 =
                    h5Path + "/" + child;

                const std::string childLogical =
                    logicalName.empty() ? child : logicalName + "/" + child;

                if (reader.file().getObjectType(childH5) == HighFive::ObjectType::Group) {
                    visit(childH5, childLogical);
                }
            }
        };

        visit(root, "");
    }

    void saveFinalOutput() const {
        H5Writer writer(params_.outname);
        writer.writeRunParams(params_);
        writer.writeModelMetadata(model_);
        writer.writeAccumulator(accumulator_);

        for (const auto& [name, acc] : arrayAccumulators_) {
            if (acc.count == 0) continue;

            const std::string base = "/array_observables/" + name;

            writer.writeArray(base, acc.mean(), acc.shape);
            writer.writeScalar(base + "_count", acc.count);
        }
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

        writeArrayAccumulatorCheckpoint(writer);
    }

    void loadCheckpoint() {
        H5Reader reader(params_.checkpointFile);

        completedSamples_ =
            reader.readInt("/checkpoint/runner/completed_samples");

        auto modelGroup =
            reader.file().getGroup("/checkpoint/replicas/0/model");

        model_.loadCheckpoint(modelGroup);

        loadAccumulatorCheckpoint(reader, accumulator_);

        loadArrayAccumulatorCheckpoint(reader);
    }

    ModelT& model_;
    RunParams params_;
    ObservableAccumulator accumulator_;

    std::unordered_map<std::string, ArrayAccumulator> arrayAccumulators_;

    int completedSamples_ = 0;
};

} // namespace mc