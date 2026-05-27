#pragma once

namespace mc {

/*
A model used with mc::Runner<ModelT> must provide:

    void runThermalization(int sweeps);

    void runBlock(const mc::BlockSpec& block);

    mc::ObservableBatch fetchObservables();

    void saveCheckpoint(mc::H5Writer& writer) const;

    void loadCheckpoint(mc::H5Reader& reader);

    void writeMetadata(HighFive::Group& group) const;

The model owns:
    - physics
    - lattice geometry
    - update rules
    - RNG state
    - model-specific checkpoint data
    - model-specific metadata

The runner owns:
    - sampling protocol
    - generic checkpoint timing
    - accumulator state
    - HDF5 final output
*/

}