#pragma once

#include "mc/Params.hpp"
#include "mc/Accumulator.hpp"

#include <highfive/H5File.hpp>

#include <string>
#include <vector>

namespace mc {

class H5Writer {
public:
    explicit H5Writer(const std::string& filename);

    template <class T>
    void writeScalar(const std::string& path, const T& value) {
        file_.createDataSet<T>(path, HighFive::DataSpace::From(value)).write(value);
    }

    template <class T>
    void writeVector(const std::string& path, const std::vector<T>& data) {
        file_.createDataSet<T>(path, HighFive::DataSpace::From(data)).write(data);
    }

    template <class ModelT>
    void writeModelMetadata(const ModelT& model) {
        auto g = getOrCreateGroup("/metadata/model");
        model.writeMetadata(g);
    }

    template <class T>
    void writeArray(
        const std::string& path,
        const std::vector<T>& data,
        const std::vector<std::size_t>& shape
    ) {
        HighFive::DataSpace space(shape);
        file_.createDataSet<T>(path, space).write(data);
    }

    void writeRunParams(const RunParams& params);
    void writeAccumulator(const ObservableAccumulator& acc);
    void writeAccumulatorCheckpoint(
        const ObservableAccumulator& acc,
        const std::string& basePath = "/checkpoint/accumulators"
    );

    HighFive::File& file() { return file_; }

private:
    HighFive::Group getOrCreateGroup(const std::string& path);

    HighFive::File file_;
};

class H5Reader {
public:
    explicit H5Reader(const std::string& filename);

    int readInt(const std::string& path) const;

    template <class T>
    T readScalar(const std::string& path) const {
        T value;
        file_.getDataSet(path).read(value);
        return value;
    }

    template <class T>
    void readVector(const std::string& path, std::vector<T>& data) const {
        file_.getDataSet(path).read(data);
    }

    HighFive::File& file() { return file_; }

private:
    HighFive::File file_;
};

}