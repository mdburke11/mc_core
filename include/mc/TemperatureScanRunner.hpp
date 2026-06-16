#pragma once

#include "mc/Params.hpp"
#include "mc/Runner.hpp"
#include "mc/Accumulator.hpp"
#include "mc/H5IO.hpp"
#include "mc/AccumulatorIO.hpp"
#include "mc/ArrayObservable.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <csignal>
#include <algorithm>
#include <cstdio>

namespace mc {

struct TemperatureScanParams {
    double T0 = 1.0;
    double Tf = 4.0;
    int NT = 16;
    std::string T_mode = "linear";
    std::string annealDirection = "auto"; // "cooling", "heating", "auto"

    static TemperatureScanParams from(const Params& p) {
        TemperatureScanParams out;
        out.T0 = p.getDouble("T0", out.T0);
        out.Tf = p.getDouble("Tf", out.Tf);
        out.NT = p.getInt("NT", out.NT);
        out.T_mode = p.getString("T_mode", out.T_mode);
        out.annealDirection = p.getString("annealDirection", out.annealDirection);
        return out;
    }
};

inline std::vector<double> makeTemperatureGrid(const TemperatureScanParams& p) {
    std::vector<double> T(p.NT);

    if (p.NT == 1) {
        T[0] = p.T0;
        return T;
    }

    if (p.T_mode == "linear") {
        for (int i = 0; i < p.NT; ++i) {
            double x = static_cast<double>(i) / (p.NT - 1);
            T[i] = p.T0 + x * (p.Tf - p.T0);
        }
    } else if (p.T_mode == "log") {
        double logT0 = std::log(p.T0);
        double logTf = std::log(p.Tf);

        for (int i = 0; i < p.NT; ++i) {
            double x = static_cast<double>(i) / (p.NT - 1);
            T[i] = std::exp(logT0 + x * (logTf - logT0));
        }
    } else {
        throw std::runtime_error("Unknown T_mode: " + p.T_mode);
    }

    if (p.annealDirection == "cooling") {
        if (T.front() < T.back()) {
            std::reverse(T.begin(), T.end());
        }
    } else if (p.annealDirection == "heating") {
        if (T.front() > T.back()) {
            std::reverse(T.begin(), T.end());
        }
    } else if (p.annealDirection == "auto") {
        // keep user-provided T0 -> Tf direction
    } else {
        throw std::runtime_error(
            "Unknown annealDirection: " + p.annealDirection
        );
    }

    return T;
}

template <class ModelT, class FactoryT>
class TemperatureScanRunner {
public:

    using TemperatureDerivedFunction = std::function<DerivedFunction(double)>;

    TemperatureScanRunner(
        FactoryT factory,
        const RunParams& runParams,
        const TemperatureScanParams& scanParams
    )
        : factory_(factory),
          runParams_(runParams),
          scanParams_(scanParams),
          temperatures_(makeTemperatureGrid(scanParams)) {}

    void addDerivedObservable(
        const std::string& name,
        TemperatureDerivedFunction f
    ) {
        derived_[name] = std::move(f);
    }

    void addDerivedToAccumulator(ObservableAccumulator& acc, double T) {
        for (const auto& [name, makeFunc] : derived_) {
            acc.addDerivedObservable(name, makeFunc(T));
        }
    }

    void run() {

        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        stopRequested.store(false);

        std::size_t ti = 0;
        bool hasResume = runParams_.resume;

        while (ti < temperatures_.size()) {

            double T = temperatures_[ti];

            ModelT model = factory_(T);
            ObservableAccumulator acc(runParams_.numBins);
            addDerivedToAccumulator(acc, T);
            std::unordered_map<std::string, ArrayAccumulator> arrayAccs;

            int completed = 0;

            if (hasResume) {
                std::size_t loadedTi = 0;
                int loadedCompleted = 0;

                // First read only the checkpoint header
                {
                    H5Reader reader(runParams_.checkpointFile);

                    loadedTi = static_cast<std::size_t>(
                        reader.readScalar<unsigned long long>(
                            "/checkpoint/runner/current_temperature_index"
                        )
                    );

                    loadedCompleted =
                        reader.readInt("/checkpoint/runner/completed_samples_at_T");
                }

                ti = loadedTi;
                T = temperatures_[ti];

                // Rebuild model and accumulator at the correct T
                model = factory_(T);
                acc = ObservableAccumulator(runParams_.numBins);
                addDerivedToAccumulator(acc, T);

                // Now load full checkpoint into correctly constructed objects
                loadCheckpoint(loadedTi, loadedCompleted, model, acc, arrayAccs);

                completed = loadedCompleted;
                hasResume = false;
            }

            std::cout << "\n=== Temperature "
                    << ti + 1 << " / " << temperatures_.size()
                    << " : T = " << T << " ===\n";

            if (completed > 0) {
                std::cout << "Continuing from sample "
                        << completed << " / "
                        << runParams_.numSamples << "\n";
            } else {
                model.runThermalization(runParams_.thermSweeps);
            }

            BlockSpec block;
            block.sweepsPerBlock = runParams_.sweepsPerBlock;
            block.measInterval = runParams_.measInterval;

            while (completed < runParams_.numSamples) {
                model.runBlock(block);

                auto batch = model.fetchObservables();
                acc.addBatch(batch);

                for (const auto& arrBatch : model.fetchArrayObservables()) {
                    arrayAccs[arrBatch.name].add(arrBatch);
                }

                completed += static_cast<int>(batch.size());

                if (completed % runParams_.checkpointInterval == 0) {
                    saveCheckpoint(ti, completed, model, acc, arrayAccs);
                }

                if (stopRequested.load()) {
                    std::cout << "Stop signal received. Saving temperature-scan checkpoint.\n";
                    saveCheckpoint(ti, completed, model, acc, arrayAccs);
                    return;
                }

                if (completed % 1000 == 0) {
                    std::cout << "Completed samples: "
                            << completed << " / "
                            << runParams_.numSamples << "\n";
                }
            }

            for (const auto& [name, data] : acc.rawData()) {
                auto s = acc.stats(name);
                means_[name].push_back(s.mean);
                errors_[name].push_back(s.error);
            }

            for (const auto& [name, f] : acc.derivedObservables()) {
                auto s = acc.derivedStats(name);
                means_[name].push_back(s.mean);
                errors_[name].push_back(s.error);
            }

            completedArrays_.push_back(std::move(arrayAccs));

            ++ti;
        }

        saveOutput(means_, errors_);
    }

private:

    void loadTempDataSoFar(H5Reader& reader) {
        if (!reader.file().exist("/checkpoint/temp_data_so_far")) {
            return;
        }

        auto g = reader.file().getGroup("/checkpoint/temp_data_so_far");

        for (const auto& name : g.listObjectNames()) {
            std::vector<double> vals;
            g.getDataSet(name).read(vals);

            const std::string suffix = "_err";

            if (
                name.size() > suffix.size() &&
                name.substr(name.size() - suffix.size()) == suffix
            ) {
                std::string base =
                    name.substr(0, name.size() - suffix.size());

                errors_[base] = std::move(vals);
            } else {
                means_[name] = std::move(vals);
            }
        }
    }

    bool loadCheckpoint(
        std::size_t& ti,
        int& completed,
        ModelT& model,
        ObservableAccumulator& acc,
        std::unordered_map<std::string, ArrayAccumulator>& arrayAccs
    ) {
        H5Reader reader(runParams_.checkpointFile);

        ti = static_cast<std::size_t>(
            reader.readScalar<unsigned long long>(
                "/checkpoint/runner/current_temperature_index"
            )
        );

        completed =
            reader.readInt("/checkpoint/runner/completed_samples_at_T");

        auto modelGroup =
            reader.file().getGroup("/checkpoint/replicas/0/model");

        model.loadCheckpoint(modelGroup);

        loadAccumulatorCheckpoint(reader, acc);
        loadTempDataSoFar(reader);
        loadCurrentArrayCheckpoint(reader, arrayAccs);
        loadCompletedArraysCheckpoint(reader);

        std::cout << "Resumed temperature scan at T index "
                << ti << ", samples "
                << completed << " / "
                << runParams_.numSamples << "\n";

        return true;
    }

    void saveCheckpoint(
        std::size_t ti,
        int completed,
        const ModelT& model,
        const ObservableAccumulator& acc,
        const std::unordered_map<std::string, ArrayAccumulator>& arrayAccs
    ) const {
        std::cout << "Saving temperature-scan checkpoint at T index "
                << ti << ", samples " << completed << "\n";

        const std::string tmp = runParams_.checkpointFile + ".tmp";

        {
            H5Writer writer(tmp);

            writer.writeScalar(
                "/checkpoint/runner/current_temperature_index",
                static_cast<unsigned long long>(ti)
            );

            writer.writeScalar(
                "/checkpoint/runner/completed_samples_at_T",
                completed
            );

            writer.writeVector("/checkpoint/runner/T_values", temperatures_);

            auto modelGroup =
                writer.file().createGroup("/checkpoint/replicas/0/model");

            model.saveCheckpoint(modelGroup);

            writer.writeAccumulatorCheckpoint(acc);

            for (const auto& [name, vals] : means_) {
                writer.writeVector(
                    "/checkpoint/temp_data_so_far/" + name,
                    vals
                );
            }

            for (const auto& [name, vals] : errors_) {
                writer.writeVector(
                    "/checkpoint/temp_data_so_far/" + name + "_err",
                    vals
                );
            }

            writeCurrentArrayCheckpoint(writer, arrayAccs);
            writeCompletedArraysCheckpoint(writer);
        }

        std::rename(tmp.c_str(), runParams_.checkpointFile.c_str());
    }

    void saveOutput(
        const std::unordered_map<std::string, std::vector<double>>& means,
        const std::unordered_map<std::string, std::vector<double>>& errors
    ) const {
        H5Writer writer(runParams_.outname);

        writer.writeRunParams(runParams_);
        writer.writeVector("/temp_data/T", temperatures_);

        for (const auto& [name, vals] : means) {
            writer.writeVector("/temp_data/" + name, vals);

            auto it = errors.find(name);
            if (it != errors.end()) {
                writer.writeVector("/temp_data/" + name + "_err", it->second);
            }
        }

        for (std::size_t i = 0; i < completedArrays_.size(); ++i) {
            bool hasArrays = false;
            for (const auto& [name, acc] : completedArrays_[i]) {
                if (acc.count > 0) { hasArrays = true; break; }
            }
            if (!hasArrays) continue;

            const std::string tbase =
                "/array_observables/T" + std::to_string(i);

            writer.writeScalar(tbase + "/T", temperatures_[i]);

            for (const auto& [name, acc] : completedArrays_[i]) {
                if (acc.count == 0) continue;
                const std::string base = tbase + "/" + name;
                writer.writeArray(base, acc.mean(), acc.shape);
                writer.writeScalar(base + "_count", acc.count);
            }
        }
    }

    void writeCurrentArrayCheckpoint(
        H5Writer& writer,
        const std::unordered_map<std::string, ArrayAccumulator>& arrayAccs
    ) const {
        for (const auto& [name, acc] : arrayAccs) {
            const std::string base = "/checkpoint/array_observables/" + name;
            writer.writeScalar(base + "/count", acc.count);
            if (acc.count > 0 && !acc.sum.empty()) {
                writer.writeArray(base + "/sum", acc.sum, acc.shape);
                std::vector<unsigned long long> shape(
                    acc.shape.begin(), acc.shape.end()
                );
                writer.writeVector(base + "/shape", shape);
            }
        }
    }

    void loadCurrentArrayCheckpoint(
        H5Reader& reader,
        std::unordered_map<std::string, ArrayAccumulator>& arrayAccs
    ) {
        const std::string root = "/checkpoint/array_observables";
        if (!reader.file().exist(root)) return;

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
                arrayAccs[logicalName] = std::move(acc);
                return;
            }
            for (const auto& child : g.listObjectNames()) {
                const std::string childH5 = h5Path + "/" + child;
                const std::string childLogical =
                    logicalName.empty() ? child : logicalName + "/" + child;
                if (reader.file().getObjectType(childH5) ==
                    HighFive::ObjectType::Group) {
                    visit(childH5, childLogical);
                }
            }
        };
        visit(root, "");
    }

    void writeCompletedArraysCheckpoint(H5Writer& writer) const {
        for (std::size_t i = 0; i < completedArrays_.size(); ++i) {
            for (const auto& [name, acc] : completedArrays_[i]) {
                if (acc.count == 0) continue;
                const std::string base =
                    "/checkpoint/array_data_so_far/T" +
                    std::to_string(i) + "/" + name;
                writer.writeScalar(base + "/count", acc.count);
                if (!acc.sum.empty()) {
                    writer.writeArray(base + "/sum", acc.sum, acc.shape);
                    std::vector<unsigned long long> shape(
                        acc.shape.begin(), acc.shape.end()
                    );
                    writer.writeVector(base + "/shape", shape);
                }
            }
        }
    }

    void loadCompletedArraysCheckpoint(H5Reader& reader) {
        const std::string root = "/checkpoint/array_data_so_far";
        if (!reader.file().exist(root)) return;

        auto g = reader.file().getGroup(root);
        int maxT = -1;
        for (const auto& child : g.listObjectNames()) {
            if (!child.empty() && child[0] == 'T') {
                try { maxT = std::max(maxT, std::stoi(child.substr(1))); }
                catch (...) {}
            }
        }
        if (maxT < 0) return;
        completedArrays_.resize(maxT + 1);

        for (int i = 0; i <= maxT; ++i) {
            const std::string troot = root + "/T" + std::to_string(i);
            if (!reader.file().exist(troot)) continue;

            std::function<void(const std::string&, const std::string&)> visit;
            visit = [&](const std::string& h5Path,
                        const std::string& logicalName) {
                auto g = reader.file().getGroup(h5Path);
                if (g.exist("count")) {
                    ArrayAccumulator acc;
                    acc.count =
                        reader.readScalar<unsigned long long>(h5Path + "/count");
                    if (acc.count > 0 &&
                        reader.file().exist(h5Path + "/sum")) {
                        reader.readArrayFlat(h5Path + "/sum", acc.sum);
                        std::vector<unsigned long long> shape;
                        reader.file().getDataSet(h5Path + "/shape").read(shape);
                        acc.shape.assign(shape.begin(), shape.end());
                    }
                    completedArrays_[i][logicalName] = std::move(acc);
                    return;
                }
                for (const auto& child : g.listObjectNames()) {
                    const std::string childH5 = h5Path + "/" + child;
                    const std::string childLogical =
                        logicalName.empty()
                            ? child
                            : logicalName + "/" + child;
                    if (reader.file().getObjectType(childH5) ==
                        HighFive::ObjectType::Group) {
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
    std::vector<double> temperatures_;

    std::unordered_map<std::string, TemperatureDerivedFunction> derived_;

    std::unordered_map<std::string, std::vector<double>> means_;
    std::unordered_map<std::string, std::vector<double>> errors_;

    std::vector<std::unordered_map<std::string, ArrayAccumulator>> completedArrays_;
};

}