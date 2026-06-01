#pragma once

#include "mc/H5IO.hpp"

namespace mc {

template <class ModelT>
auto writeExtraObservablesIfAvailable(
    const ModelT& model,
    H5Writer& writer,
    int
) -> decltype(model.writeExtraObservables(writer), void()) {
    model.writeExtraObservables(writer);
}

template <class ModelT>
void writeExtraObservablesIfAvailable(
    const ModelT&,
    H5Writer&,
    long
) {}

}