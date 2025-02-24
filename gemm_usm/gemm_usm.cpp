#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "oneapi/math.hpp"


template <typename fp>
inline fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}

template <typename vec>
inline void rand_matrix(vec& M, oneapi::math::transpose trans, int m, int n, int ld) {
    using fp = typename vec::value_type;

    if (trans == oneapi::math::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M.at(i + j * ld) = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M.at(j + i * ld) = rand_scalar<fp>();
    }
}

template <typename T>
inline void print_2x2_matrix_values(T M, int ldM, std::string M_name) {
    std::cout << std::endl;
    std::cout << "\t\t\t" << M_name << " = [ " << M[0 * ldM + 0] << ", " << M[1 * ldM + 0]
              << ", ...\n";
    std::cout << "\t\t\t    [ " << M[0 * ldM + 1] << ", " << M[1 * ldM + 1] << ", ...\n";
    std::cout << "\t\t\t    [ "
              << "...\n";
    std::cout << std::endl;
}


void run_gemm_example(const sycl::device& dev) {
    oneapi::math::transpose transA = oneapi::math::transpose::trans;
    oneapi::math::transpose transB = oneapi::math::transpose::nontrans;

    // matrix data sizes
    const int m = 45;
    const int n = 98;
    const int k = 67;

    // leading dimensions of data
    const int ldA = 103;
    const int ldB = 105;
    const int ldC = 106;
    const int sizea = (transA == oneapi::math::transpose::nontrans) ? ldA * k : ldA * m;
    const int sizeb = (transB == oneapi::math::transpose::nontrans) ? ldB * n : ldB * k;
    const int sizec = ldC * n;

    // set scalar fp values
    float alpha = 2.0f;
    float beta = 3.0f;

    // Catch asynchronous exceptions for CPU and GPU
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception during GEMM:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    //
    // Data Preparation on host
    //
    std::vector<float> A(sizea);
    std::vector<float> B(sizeb);
    std::vector<float> C(sizec);
    std::vector<float> result_cpu(sizec);
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(C.begin(), C.end(), 0);
    std::fill(result_cpu.begin(), result_cpu.end(), 0);

    rand_matrix(A, transA, m, k, ldA);
    rand_matrix(B, transB, k, n, ldB);
    rand_matrix(C, oneapi::math::transpose::nontrans, m, n, ldC);

    //
    // Preparation
    //
    sycl::queue queue(dev, exception_handler);
    sycl::context context = queue.get_context();

    // allocate on CPU device and copy data from host to SYCL CPU device
    float* cpu_A = sycl::malloc_device<float>(sizea * sizeof(float), queue);
    float* cpu_B = sycl::malloc_device<float>(sizeb * sizeof(float), queue);
    float* cpu_C = sycl::malloc_device<float>(sizec * sizeof(float), queue);
    if (!cpu_A || !cpu_B || !cpu_C) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }
    queue.memcpy(cpu_A, A.data(), sizea * sizeof(float)).wait();
    queue.memcpy(cpu_B, B.data(), sizeb * sizeof(float)).wait();
    queue.memcpy(cpu_C, C.data(), sizec * sizeof(float)).wait();

    //
    // Execute Gemm 
    //
    // add oneapi::math::blas::gemm to execution queue
    auto gemm_event = oneapi::math::blas::column_major::gemm(
        queue, transA, transB,
        m, n, k, alpha, cpu_A, ldA, cpu_B, ldB, beta, cpu_C, ldC);

    // Wait until calculations are done
    gemm_event.wait_and_throw();

    // copy data to host
    queue.memcpy(result_cpu.data(), cpu_C, sizec * sizeof(float)).wait_and_throw();

    // print results
    std::cout << "\n\t\tGEMM parameters:" << std::endl;
    std::cout << "\t\t\ttransA = "
              << (transA == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (transA == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << ", transB = "
              << (transB == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (transB == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << ", k = " << k << std::endl;
    std::cout << "\t\t\tlda = " << ldA << ", ldB = " << ldB << ", ldC = " << ldC << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;

    std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(A.data(), ldA, "A");

    // output the top 2x2 block of B matrix
    print_2x2_matrix_values(B.data(), ldB, "B");

    // output the top 2x2 block of C matrix
    print_2x2_matrix_values(result_cpu.data(), ldC, "C");

    sycl::free(cpu_C, queue);
    sycl::free(cpu_B, queue);
    sycl::free(cpu_A, queue);
}

int main(int argc, char** argv) {

    try {
        /* Specity device */ 
        sycl::device dev; // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
        // sycl::device dev(sycl::cpu_selector_v); // intel CPU
        // sycl::device dev(sycl::gpu_selector_v); // intel GPU/iGPU
        // sycl::device dev([](const sycl::device &dev) {
        //     return dev.is_gpu() && (dev.get_info<sycl::info::device::vendor_id>() == NVIDIA_ID);
        // }); // NVIDIA GPU

        std::cout << "Running BLAS GEMM USM example" << std::endl;
        std::cout << "Running with single precision real data type on:" << std::endl;
        std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
        run_gemm_example(dev);
        std::cout << "BLAS GEMM USM example ran OK" << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception during GEMM:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        std::cerr << "Caught std::exception during GEMM:";
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }
    return 0;
}
