#include "bitonic_sort.hpp"

#include <chrono>
#include <random>
#include <thread>


int main(int argc, char* argv[]) {

    using DATA_TYPE = float;
    const size_t DATA_SIZE = 1000000;

    // random data
    std::vector<DATA_TYPE> data(DATA_SIZE);
    std::mt19937 gen;
    std::uniform_real_distribution<DATA_TYPE> dist(1.0, 100.0);

    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = dist(gen);
    }

    std::vector<DATA_TYPE> sorted_data;
    std::vector<DATA_TYPE> std_sorted;

    const size_t LOOPS = 10;

    // std::sort
    double std_time = 0.0;
    for (size_t i = 0; i < LOOPS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std_sorted = data;
        std::sort(std_sorted.begin(), std_sorted.end());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        std_time += elapsed.count();
    }
    std_time /= LOOPS;

    // bitonic_sort
    double bitonic_time = 0.0;
    sycl::queue q;
    for (size_t i = 0; i < LOOPS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        bitonic_sort<DATA_TYPE>(q, data, sorted_data);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        bitonic_time += elapsed.count();
    }
    bitonic_time /= LOOPS;

    // verification
    bool correct = true;
    for (size_t i = 0; i < DATA_SIZE; i++) {
        if (std::abs(sorted_data[i] - std_sorted[i]) > std::numeric_limits<DATA_TYPE>::epsilon()) {
            correct = false;
            break;
        }
    }
    std::cout << (correct ? "SUCCESS" : "FAILED") << std::endl;

    std::cout << "DEVICE: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "DATA_SIZE: " << DATA_SIZE << std::endl;

    std::cout << "std::sort: " << std_time << " us, bitonic_sort: " << bitonic_time << " us" <<std::endl;

    return 0;
}
