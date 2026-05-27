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

        auto modelName = params.getString("model", "Ising2D");
        if (modelName != "Ising2D") {
            throw std::runtime_error("This executable only supports model = Ising2D");
        }

        params.validateUnused();

        Ising2D model(modelParams);

        mc::Runner<Ising2D> runner(model, runParams);


        const int N = modelParams.L * modelParams.L;
        const double beta = 1.0 / modelParams.T;

        runner.addDerivedObservable("Cv", [N, beta](const auto& m) {
            return N * beta * beta * (m.at("E2") - m.at("E") * m.at("E"));
        });

        runner.addDerivedObservable("chi_abs", [N, beta](const auto& m) {
            return N * beta * (m.at("M2") - m.at("absM") * m.at("absM"));
        });


        runner.run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}