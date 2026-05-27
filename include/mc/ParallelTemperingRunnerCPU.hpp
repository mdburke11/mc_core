#pragma once

#include "mc/Params.hpp"
#include "mc/Accumulator.hpp"
#include "mc/Runner.hpp"
#include "mc/H5IO.hpp"
#include "mc/TemperatureScanRunner.hpp"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

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

        if (runParams_.resume) {
            // We can add resume after first successful non-resume PT test.
            std::cout << "PT resume not implemented yet in this first pass.\n";
        } else {
            thermalize();
        }

        int completed = 0;
        BlockSpec block;
        block.sweepsPerBlock = runParams_.sweepsPerBlock;
        block.measInterval = runParams_.measInterval;

        while (completed < runParams_.numSamples) {
            runAllReplicas(block);

            measureAllTemperatureSlots();

            completed += 1;

            if (completed % ptParams_.exchInterval == 0) {
                attemptSwaps(completed);
            }

            if (completed % runParams_.checkpointInterval == 0) {
                saveCheckpoint(completed);
            }

            if (stopRequested.load()) {
                std::cout << "Stop signal received. Saving PT checkpoint.\n";
                saveCheckpoint(completed);
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

    void measureAllTemperatureSlots() {
        const int NT = static_cast<int>(tempSlots_.size());

        for (int t = 0; t < NT; ++t) {
            int r = temp_to_replica_[t];

            auto batch = replicas_[r].model.fetchObservables();
            tempSlots_[t].acc.addBatch(batch);
        }
    }

    void attemptSwaps(int completed) {
        const int NT = static_cast<int>(tempSlots_.size());

        int parity = (completed / ptParams_.exchInterval) % 2;
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
    }

    void saveCheckpoint(int completed) const {
        std::cout << "Saving PT checkpoint at sample "
                  << completed << "\n";

        H5Writer writer(runParams_.checkpointFile);

        writer.writeScalar(
            "/checkpoint/runner/completed_samples",
            completed
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

        // Accumulator checkpointing per temperature slot comes next.
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
};

}