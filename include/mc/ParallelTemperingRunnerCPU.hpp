#pragma once

#include "mc/Params.hpp"
#include "mc/Accumulator.hpp"
#include "mc/Runner.hpp"
#include "mc/H5IO.hpp"
#include "mc/TemperatureScanRunner.hpp"
#include "mc/AccumulatorIO.hpp"
#include "mc/ArrayObservable.hpp"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <highfive/H5File.hpp>

#ifdef MC_HAS_OPENMP
#include <omp.h>
#endif

namespace mc {

struct ParallelTemperingParams {
    int exchInterval = 1;
    int swapSeed = 12345;

    static ParallelTemperingParams from(const Params& p) {
        ParallelTemperingParams out;
        out.exchInterval = p.getInt("exchInterval", out.exchInterval);
        out.swapSeed = p.getInt("swapSeed", out.swapSeed);
        return out;
    }
};

template <class ModelT>
struct PTReplicaCPU {
    ModelT model;
    int tempIndex = 0;

    PTReplicaCPU(ModelT&& model_, int tempIndex_)
        : model(std::move(model_)),
          tempIndex(tempIndex_) {}
};

struct PTTempSlot {
    double T = 1.0;
    double beta = 1.0;
    ObservableAccumulator acc;

    std::unordered_map<std::string, ArrayAccumulator> arrays;

    void addArrayBatch(const ArrayObservableBatch& batch) {
        arrays[batch.name].add(batch);
    }

    PTTempSlot(double T_, int numBins)
        : T(T_),
          beta(1.0 / T_),
          acc(numBins) {}
};



template <class ModelT, class FactoryT>
class ParallelTemperingRunnerCPU {
public:
    using TemperatureDerivedFunction =
        std::function<DerivedFunction(double)>;

    ParallelTemperingRunnerCPU(
        FactoryT factory,
        const RunParams& runParams,
        const TemperatureScanParams& scanParams,
        const ParallelTemperingParams& ptParams
    )
        : factory_(factory),
          runParams_(runParams),
          scanParams_(scanParams),
          ptParams_(ptParams),
          temperatures_(makeTemperatureGrid(scanParams)),
          rng_(ptParams.swapSeed) {}

    void addDerivedObservable(
        const std::string& name,
        TemperatureDerivedFunction f
    ) {
        derived_[name] = std::move(f);
    }

    void run() {
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        stopRequested.store(false);

        initialize();

        int completed = 0;
        int ptStep = 0;

        if (runParams_.resume) {
            loadCheckpoint(completed, ptStep);

            if (completed >= runParams_.numSamples) {
                std::cout
                    << "Checkpoint already satisfies requested "
                    << "numSamples.\n";

                saveOutput();
                return;
            }
        }
        else {
            thermalize();
        }

        // Make sure the next checkpoint is ahead of the resumed sample count.
        nextCheckpoint = runParams_.checkpointInterval;
        while (nextCheckpoint <= completed) {
            nextCheckpoint += runParams_.checkpointInterval;
        }

        BlockSpec block;
        block.sweepsPerBlock = runParams_.sweepsPerBlock;
        block.measInterval = runParams_.measInterval;

        while (completed < runParams_.numSamples) {
            runAllReplicas(block);

            std::size_t n = measureAllTemperatureSlots();
            completed += static_cast<int>(n);

            ptStep++;

            if (ptStep % ptParams_.exchInterval == 0) {
                attemptSwaps(ptStep);
            }

            if (completed >= nextCheckpoint) {
                saveCheckpoint(completed, ptStep);

                while (nextCheckpoint <= completed) {
                    nextCheckpoint += runParams_.checkpointInterval;
                }
            }

            if (stopRequested.load()) {
                std::cout << "Stop signal received. Saving PT checkpoint.\n";
                saveCheckpoint(completed, ptStep);
                return;
            }

            if (completed % 1000 == 0) {
                std::cout << "Completed PT samples: "
                          << completed << " / "
                          << runParams_.numSamples << "\n";
            }
        }

        saveOutput();
    }

private:
    void initialize() {
        const int NT = static_cast<int>(temperatures_.size());

        replicas_.clear();
        tempSlots_.clear();

        temp_to_replica_.resize(NT);
        replica_to_temp_.resize(NT);

        swapAttempts_.assign(std::max(0, NT - 1), 0);
        swapAccepts_.assign(std::max(0, NT - 1), 0);

        for (int t = 0; t < NT; ++t) {
            double T = temperatures_[t];

            ModelT model = factory_(T);
            model.setTemperature(T);

            replicas_.emplace_back(std::move(model), t);
            tempSlots_.emplace_back(T, runParams_.numBins);

            for (const auto& [name, makeFunc] : derived_) {
                tempSlots_.back().acc.addDerivedObservable(
                    name,
                    makeFunc(T)
                );
            }

            temp_to_replica_[t] = t;
            replica_to_temp_[t] = t;
        }
    }

    void thermalize() {
        std::cout << "Thermalizing PT replicas for "
                  << runParams_.thermSweeps << " sweeps\n";

        #ifdef MC_HAS_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int r = 0; r < static_cast<int>(replicas_.size()); ++r) {
            replicas_[r].model.runThermalization(runParams_.thermSweeps);
        }
    }

    void runAllReplicas(const BlockSpec& block) {
        #ifdef MC_HAS_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int r = 0; r < static_cast<int>(replicas_.size()); ++r) {
            replicas_[r].model.runBlock(block);
        }
    }

    std::size_t measureAllTemperatureSlots() {
        const int NT = static_cast<int>(tempSlots_.size());
        std::size_t n = 0;

        for (int t = 0; t < NT; ++t) {
            int r = temp_to_replica_[t];

            auto batch = replicas_[r].model.fetchObservables();

            if (t == 0) {
                n = batch.size();
            }

            tempSlots_[t].acc.addBatch(batch);

            for (const auto& arrBatch : replicas_[r].model.fetchArrayObservables()) {
                tempSlots_[t].addArrayBatch(arrBatch);
            }
        }

        return n;
    }

    void attemptSwaps(int ptStep) {
        const int NT = static_cast<int>(tempSlots_.size());

        int parity = (ptStep / ptParams_.exchInterval) % 2;
        int start = parity;

        std::uniform_real_distribution<double> uni(0.0, 1.0);

        for (int t = start; t + 1 < NT; t += 2) {
            int r1 = temp_to_replica_[t];
            int r2 = temp_to_replica_[t + 1];

            double beta1 = tempSlots_[t].beta;
            double beta2 = tempSlots_[t + 1].beta;

            double E1 = replicas_[r1].model.energy();
            double E2 = replicas_[r2].model.energy();

            double logAccept = (beta1 - beta2) * (E1 - E2);

            swapAttempts_[t] += 1;

            bool accept =
                logAccept >= 0.0 ||
                uni(rng_) < std::exp(logAccept);

            if (accept) {
                swapAccepts_[t] += 1;

                std::swap(temp_to_replica_[t],
                          temp_to_replica_[t + 1]);

                replica_to_temp_[r1] = t + 1;
                replica_to_temp_[r2] = t;

                replicas_[r1].tempIndex = t + 1;
                replicas_[r2].tempIndex = t;

                replicas_[r1].model.setTemperature(tempSlots_[t + 1].T);
                replicas_[r2].model.setTemperature(tempSlots_[t].T);
            }
        }
    }

    void saveOutput() const {
        H5Writer writer(runParams_.outname);

        writer.writeRunParams(runParams_);
        writer.writeVector("/temp_data/T", temperatures_);

        std::unordered_map<std::string, std::vector<double>> means;
        std::unordered_map<std::string, std::vector<double>> errors;

        for (int t = 0; t < static_cast<int>(tempSlots_.size()); ++t) {
            const auto& acc = tempSlots_[t].acc;

            for (const auto& [name, data] : acc.rawData()) {
                auto s = acc.stats(name);
                means[name].push_back(s.mean);
                errors[name].push_back(s.error);
            }

            for (const auto& [name, f] : acc.derivedObservables()) {
                auto s = acc.derivedStats(name);
                means[name].push_back(s.mean);
                errors[name].push_back(s.error);
            }
        }

        for (const auto& [name, vals] : means) {
            writer.writeVector("/temp_data/" + name, vals);

            auto it = errors.find(name);
            if (it != errors.end()) {
                writer.writeVector(
                    "/temp_data/" + name + "_err",
                    it->second
                );
            }
        }

        writer.writeVector("/pt/swap_attempts", swapAttempts_);
        writer.writeVector("/pt/swap_accepts", swapAccepts_);

        for (int t = 0; t < static_cast<int>(tempSlots_.size()); ++t) {
            const auto& slot = tempSlots_[t];

            bool hasArrays = false;
            for (const auto& [name, acc] : slot.arrays) {
                if (acc.count > 0) {
                    hasArrays = true;
                    break;
                }
            }

            if (!hasArrays) {
                continue;
            }

            const std::string tbase =
                "/array_observables/T" + std::to_string(t);

            writer.writeScalar(tbase + "/T", slot.T);

            for (const auto& [name, acc] : slot.arrays) {
                if (acc.count == 0) continue;

                const std::string base = tbase + "/" + name;

                writer.writeArray(base, acc.mean(), acc.shape);
                writer.writeScalar(base + "_count", acc.count);
            }
        }
    }

    void saveCheckpoint(int completed, int ptStep) const {
        std::cout << "Saving PT checkpoint at sample "
                  << completed << "\n";

        const std::string tmp = runParams_.checkpointFile + ".tmp";

        {
            H5Writer writer(tmp);

            writer.writeScalar(
                "/checkpoint/runner/completed_samples",
                completed
            );

            writer.writeScalar(
                "/checkpoint/runner/pt_step",
                ptStep
            );

            writer.writeVector(
                "/checkpoint/runner/temp_to_replica",
                temp_to_replica_
            );

            writer.writeVector(
                "/checkpoint/runner/replica_to_temp",
                replica_to_temp_
            );

            writer.writeVector(
                "/checkpoint/runner/swap_attempts",
                swapAttempts_
            );

            writer.writeVector(
                "/checkpoint/runner/swap_accepts",
                swapAccepts_
            );

            writer.writeVector(
                "/checkpoint/runner/T_values",
                temperatures_
            );

            for (int r = 0; r < static_cast<int>(replicas_.size()); ++r) {
                auto g = writer.file().createGroup(
                    "/checkpoint/replicas/" +
                    std::to_string(r) +
                    "/model"
                );

                replicas_[r].model.saveCheckpoint(g);
            }

            for (int t = 0; t < static_cast<int>(tempSlots_.size()); ++t) {
                writer.writeAccumulatorCheckpoint(
                    tempSlots_[t].acc,
                    "/checkpoint/temp_slots/" +
                    std::to_string(t) +
                    "/accumulators"
                );
            }

            writeArrayAccumulatorCheckpoint(writer);

            std::ostringstream ss;
            ss << rng_;
            writer.writeScalar("/checkpoint/runner/swap_rng_state", ss.str());
        }

        std::rename(tmp.c_str(), runParams_.checkpointFile.c_str());
    }

    void loadCheckpoint(int& completed, int& ptStep) {
        H5Reader reader(runParams_.checkpointFile);

        completed =
            reader.readInt("/checkpoint/runner/completed_samples");

        ptStep =
            reader.readInt("/checkpoint/runner/pt_step");

        reader.readVector(
            "/checkpoint/runner/temp_to_replica",
            temp_to_replica_
        );

        reader.readVector(
            "/checkpoint/runner/replica_to_temp",
            replica_to_temp_
        );

        reader.readVector(
            "/checkpoint/runner/swap_attempts",
            swapAttempts_
        );

        reader.readVector(
            "/checkpoint/runner/swap_accepts",
            swapAccepts_
        );

        // --------------------
        // Replica model states
        // --------------------

        for (int r = 0; r < static_cast<int>(replicas_.size()); ++r) {

            auto g =
                reader.file().getGroup(
                    "/checkpoint/replicas/" +
                    std::to_string(r) +
                    "/model"
                );

            replicas_[r].model.loadCheckpoint(g);
        }

        // --------------------
        // Temperature-slot accumulators
        // --------------------

        for (int t = 0; t < static_cast<int>(tempSlots_.size()); ++t) {

            loadAccumulatorCheckpoint(
                reader,
                tempSlots_[t].acc,
                "/checkpoint/temp_slots/" +
                std::to_string(t) +
                "/accumulators"
            );
        }

        // --------------------
        // Restore temperatures
        // --------------------

        for (int r = 0; r < static_cast<int>(replicas_.size()); ++r) {

            int t = replica_to_temp_[r];

            replicas_[r].tempIndex = t;
            replicas_[r].model.setTemperature(
                tempSlots_[t].T
            );
        }

        loadArrayAccumulatorCheckpoint(reader);

        std::string rngState;
        reader.file().getDataSet("/checkpoint/runner/swap_rng_state").read(rngState);

        std::istringstream ss(rngState);
        ss >> rng_;



        std::cout
            << "Resumed PT checkpoint at sample "
            << completed
            << " / "
            << runParams_.numSamples
            << "\n";
    }
    
    void writeArrayAccumulatorCheckpoint(H5Writer& writer) const {
        for (int t = 0; t < static_cast<int>(tempSlots_.size()); ++t) {
            const auto& slot = tempSlots_[t];

            for (const auto& [name, acc] : slot.arrays) {
                const std::string base =
                    "/checkpoint/array_observables/T" +
                    std::to_string(t) + "/" + name;

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
    }

    void loadArrayAccumulatorCheckpoint(H5Reader& reader) {
        const std::string root = "/checkpoint/array_observables";

        if (!reader.file().exist(root)) {
            return;
        }

        for (int t = 0; t < static_cast<int>(tempSlots_.size()); ++t) {
            const std::string troot = root + "/T" + std::to_string(t);

            if (!reader.file().exist(troot)) {
                continue;
            }

            std::function<void(const std::string&, const std::string&)> visit;

            visit = [&](const std::string& h5Path, const std::string& logicalName) {
                auto g = reader.file().getGroup(h5Path);

                if (g.exist("count")) {
                    ArrayAccumulator acc;

                    acc.count =
                        reader.readScalar<unsigned long long>(h5Path + "/count");

                    if (acc.count > 0 && reader.file().exist(h5Path + "/sum")) {
                        reader.readArrayFlat(h5Path + "/sum", acc.sum);

                        std::vector<unsigned long long> shape;
                        reader.file().getDataSet(h5Path + "/shape").read(shape);

                        acc.shape.assign(shape.begin(), shape.end());
                    }

                    tempSlots_[t].arrays[logicalName] = std::move(acc);
                    return;
                }

                for (const auto& child : g.listObjectNames()) {
                    const std::string childH5 = h5Path + "/" + child;
                    const std::string childLogical =
                        logicalName.empty() ? child : logicalName + "/" + child;

                    if (
                        reader.file().getObjectType(childH5) ==
                        HighFive::ObjectType::Group
                    ) {
                        visit(childH5, childLogical);
                    }
                }
            };

            visit(troot, "");
        }
    }

    FactoryT factory_;
    RunParams runParams_;
    TemperatureScanParams scanParams_;
    ParallelTemperingParams ptParams_;

    std::vector<double> temperatures_;

    std::vector<PTReplicaCPU<ModelT>> replicas_;
    std::vector<PTTempSlot> tempSlots_;

    std::vector<int> temp_to_replica_;
    std::vector<int> replica_to_temp_;

    std::vector<unsigned long long> swapAttempts_;
    std::vector<unsigned long long> swapAccepts_;

    std::unordered_map<std::string, TemperatureDerivedFunction> derived_;

    std::mt19937_64 rng_;

    int nextCheckpoint = 0;
};

}