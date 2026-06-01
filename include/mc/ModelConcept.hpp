#pragma once

namespace mc {

/*
A model used with mc::Runner<ModelT> must provide:

    void runThermalization(int sweeps);

    void runBlock(const mc::BlockSpec& block);

    mc::ObservableBatch fetchObservables();

    void saveCheckpoint(HighFive::Group& group) const;

    void loadCheckpoint(const HighFive::Group& group);

    void writeMetadata(HighFive::Group& group) const;

The model owns:
    - physics
    - lattice geometry
    - update rules
    - RNG state
    - model-specific checkpoint data
    - model-specific metadata

PT-compatible models need to have 
double energy() const;
void setTemperature(double T);

The runner owns:
    - sampling protocol
    - generic checkpoint timing
    - accumulator state
    - HDF5 final output
*/

}