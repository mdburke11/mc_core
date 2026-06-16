#pragma once

#include "mc/Params.hpp"
#include "mc/Accumulator.hpp"

#include <highfive/H5File.hpp>

#include <string>
#include <vector>
#include <stdexcept>

namespace mc {

class H5Writer {
public:
    explicit H5Writer(const std::string& filename);

    template <class T>
    void writeScalar(const std::string& path, const T& value) {
        getOrCreateGroup(parentPath(path));
        file_.createDataSet<T>(path, HighFive::DataSpace::From(value)).write(value);
    }

    template <class T>
    void writeVector(const std::string& path, const std::vector<T>& data) {
        getOrCreateGroup(parentPath(path));
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
        getOrCreateGroup(parentPath(path));

        std::size_t expected = 1;
        for (auto n : shape) {
            expected *= n;
        }

        if (expected != data.size()) {
            throw std::runtime_error(
                "writeArray shape does not match flattened data size for path: " + path
            );
        }

        HighFive::DataSpace space(shape);
        auto dset = file_.createDataSet<T>(path, space);

        dset.write_raw(data.data());
    }

    void writeRunParams(const RunParams& params);
    void writeAccumulator(const ObservableAccumulator& acc);
    void writeAccumulatorCheckpoint(
        const ObservableAccumulator& acc,
        const std::string& basePath = "/checkpoint/accumulators"
    );

    HighFive::File& file() { return file_; }

private:

    std::string parentPath(const std::string& path) const {
        auto pos = path.find_last_of('/');
        if (pos == std::string::npos || pos == 0) {
            return "/";
        }
        return path.substr(0, pos);
    }

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

    template <class T>
    void readArrayFlat(const std::string& path, std::vector<T>& data) const {
        auto dset = file_.getDataSet(path);
        data.resize(dset.getSpace().getElementCount());
        dset.read_raw<T>(data.data());
    }

    HighFive::File& file() { return file_; }

private:
    HighFive::File file_;
};

}