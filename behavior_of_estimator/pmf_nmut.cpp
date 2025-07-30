#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <string>

#include <numeric>
#include <cassert>


// Convert r1 to q
double r1_to_q(int k, double r1) {
    return 1.0 - std::pow(1.0 - r1, k);
}

// Expected number of mutated positions
double exp_n_mutated(int L, int k, double r1) {
    double q = r1_to_q(k, r1);
    return L * q;
}

// Variance of number of mutated positions
double var_n_mutated(int L, int k, double r1, double q = -1.0) {
    if (r1 == 0.0) return 0.0;

    if (q < 0.0)
        q = r1_to_q(k, r1);

    double varN =
        L * (1.0 - q) * (q * (2.0 * k + 2.0 / r1 - 1.0) - 2.0 * k)
        + k * (k - 1.0) * std::pow(1.0 - q, 2)
        + (2.0 * (1.0 - q) / (r1 * r1)) * ((1.0 + (k - 1.0) * (1.0 - q)) * r1 - q);

    assert(varN >= 0.0);
    return varN;
}


// Compute normal CDF
double normal_cdf(double x, double mu, double sigma) {
    return 0.5 * (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2.0))));
}


// Main PDF approximation function
std::vector<double> get_nmut_pdf_using_normal_distribution(int L, int k, double r1) {
    double mu = exp_n_mutated(L, k, r1);
    double sigma = std::sqrt(var_n_mutated(L, k, r1));

    std::vector<double> pdf(L + 1, 0.0);
    for (int i = 0; i <= L; ++i) {
        double upper = normal_cdf(i + 0.5, mu, sigma);
        double lower = normal_cdf(i - 0.5, mu, sigma);
        pdf[i] = upper - lower;
    }

    // Normalize
    double sum = std::accumulate(pdf.begin(), pdf.end(), 0.0);
    if (sum > 0.0) {
        for (double& val : pdf) val /= sum;
    }

    return pdf;
}


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

    std::cout << "This program computes expectation of L-Nmut raised to power of 1/k\n";

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
    std::cout << "L = " << L << ", ";
    std::cout << "k = " << k << ", ";
    std::cout << "p = " << p << ".\n";
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

    // compute expectation of (L-Nmut)^(1/k) from the PMF
    double expectation = 0.0;
    for (int i = 0; i <= L; ++i) {
        expectation += result[i] * pow(L-i, 1.0 / k);
    }
    // Print the expectation
    std::cout << "Expectation using exhaustive PMF computation = " << expectation << "\n";

    // Write the PMF to the output file
    std::ofstream output_file(output_filename);
    if (!output_file) {
        std::cerr << "Error opening output file: " << output_filename << "\n";
        return 1;
    }

    // write the expectation of (L-Nmut)^(1/k) to the output file
    output_file << expectation << "\n";

    // close the output file
    output_file.close();


    // compute the PMF using normal distribution approximation
    std::vector<double> pdf_normal = get_nmut_pdf_using_normal_distribution(L, k, p);
    std::cout << "PMF using normal distribution approximation has been computed.\n";

    // show sum of PMF
    double sum_normal = std::accumulate(pdf_normal.begin(), pdf_normal.end(), 0.0);
    std::cout << "Sum of PMF using normal distribution approximation = " << sum_normal << "\n";

    // compute expectation of (L-Nmut)^(1/k) from the PMF
    double expectation_normal = 0.0;
    for (int i = 0; i <= L; ++i) {
        expectation_normal += pdf_normal[i] * pow(L - i, 1.0 / k);
    }
    std::cout << "Expectation using normal approximation = " << expectation_normal << "\n";


    return 0;
}
