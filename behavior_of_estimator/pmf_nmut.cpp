#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <string>

// Function to initialize the first two rows like perform_exhaustive_counting_modified
std::vector<std::vector<double>> perform_exhaustive_counting_modified(int k, double p) {
    std::vector<std::vector<double>> arr(2, std::vector<double>(k + 1, 0.0));
    arr[0][k] = pow(1 - p, k);
    for (int i = 1; i <= k; ++i) {
        arr[1][k - i] = p * pow(1 - p, k - i);
    }
    return arr;
}

std::vector<double> get_nmut_pdf_modified(int L, int k, double p) {
    auto arr = perform_exhaustive_counting_modified(k, p);

    // Allocate double buffers
    std::vector<std::vector<double>> buffers[2] = {
        std::vector<std::vector<double>>(L + 1, std::vector<double>(k + 1, 0.0)),
        std::vector<std::vector<double>>(L + 1, std::vector<double>(k + 1, 0.0))
    };

    int curr_idx = 0;
    buffers[curr_idx][0] = arr[0];
    buffers[curr_idx][1] = arr[1];

    for (int l = k + 1; l <= L; ++l) {
        auto& curr = buffers[curr_idx];
        auto& next = buffers[1 - curr_idx];

        // Compute row sums
        std::vector<double> row_sums(L + 1, 0.0);
        for (int i = 0; i <= L; ++i)
            for (int z = 0; z <= k; ++z)
                row_sums[i] += curr[i][z];

        for (int x = 0; x < L; ++x) {
            double total = (x > 0) ? row_sums[x - 1] : 0.0;
            next[x][0] = total * p;
            if (x > 0) {
                for (int z = 1; z < k; ++z) {
                    next[x][z] = curr[x - 1][z - 1] * (1 - p);
                }
            }
            next[x][k] = (curr[x][k - 1] + curr[x][k]) * (1 - p);
        }

        curr_idx = 1 - curr_idx;
    }

    // Sum over last buffer
    std::vector<double> result(L + 1, 0.0);
    for (int x = 0; x <= L; ++x)
        for (int z = 0; z <= k; ++z)
            result[x] += buffers[curr_idx][x][z];

    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: ./nmut_pdf L k p output_filename\n";
        return 1;
    }

    int L = std::atoi(argv[1]);
    int k = std::atoi(argv[2]);
    double p = std::atof(argv[3]);
    std::string output_filename = argv[4];

    // show the arguments
    std::cout << "Arguments:\n";
    std::cout << "L = " << L << "\n";
    std::cout << "k = " << k << "\n";
    std::cout << "p = " << p << "\n";
    std::cout << "Output filename = " << output_filename << "\n";

    // Validate arguments
    if (L <= 0 || k <= 0 || k >= L || p < 0.0 || p > 1.0) {
        std::cerr << "Invalid arguments. L and k must be > 0, and 0 <= p <= 1.\n";
        return 1;
    }

    // Compute the PMF
    std::vector<double> result = get_nmut_pdf_modified(L+k-1, k, p);
    std::cout << "PMF has been computed.\n";
    std::cout << std::fixed << std::setprecision(6);

    // compute expectation of X^(1/k) from the PMF
    double expectation = 0.0;
    for (int i = 0; i <= L; ++i) {
        expectation += result[i] * pow(i, 1.0 / k);
    }
    std::cout << "Expectation = " << expectation << "\n";

    // Write the PMF to the output file
    std::ofstream output_file(output_filename);
    if (!output_file) {
        std::cerr << "Error opening output file: " << output_filename << "\n";
        return 1;
    }

    // write the expectation of X^(1/k) to the output file
    output_file << expectation << "\n";

    // close the output file
    output_file.close();

    return 0;
}
