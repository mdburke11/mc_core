#include "mc/Params.hpp"
#include "mc/Runner.hpp"

#include "Ising2D.hpp"

#include <iostream>
#include <exception>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ising2d input.param\n";
        return 1;
    }

    try {
        mc::Params params(argv[1]);

        mc::RunParams runParams = mc::RunParams::from(params);
        Ising2DParams modelParams = Ising2DParams::from(params);

        Ising2D model(modelParams);

        mc::Runner<Ising2D> runner(model, runParams);
        runner.run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}