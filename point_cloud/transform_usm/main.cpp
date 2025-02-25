#include <Eigen/Dense>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
// #include "oneapi/math.hpp"

// using PointType = double; // not work with 11th Intel iGPU
using PointType = float; // work with iGPU
using PointXYZ = Eigen::Vector4<PointType>;
using PointCloudXYZ = std::vector<PointXYZ, Eigen::aligned_allocator<PointXYZ>>;

inline PointCloudXYZ transform_sycl(sycl::queue q, const PointCloudXYZ& points, Eigen::Matrix4<PointType> trans) {
    
    const size_t N = points.size();
    
    // allocate device memory
    PointType* dev_data = sycl::malloc_device<PointType>(N * 4, q);
    PointType* dev_trans = sycl::malloc_device<PointType>(16, q);
    PointType* dev_result = sycl::malloc_device<PointType>(N * 4, q);
    q.memcpy(dev_data, points[0].data(), N * 4 * sizeof(PointType)).wait();
    q.memcpy(dev_trans, trans.data(), 16 * sizeof(PointType)).wait();
    q.fill(dev_result, (PointType)1.0, N * 4);

    /* Transform */
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            // Eigen::Matrix is column major
            const auto x = dev_data[i * 4 + 0];
            const auto y = dev_data[i * 4 + 1];
            const auto z = dev_data[i * 4 + 2];
            dev_result[i * 4 + 0] = dev_trans[0] * x + dev_trans[4] * y + dev_trans[8] * z + dev_trans[12];
            dev_result[i * 4 + 1] = dev_trans[1] * x + dev_trans[5] * y + dev_trans[9] * z + dev_trans[13];
            dev_result[i * 4 + 2] = dev_trans[2] * x + dev_trans[6] * y + dev_trans[10] * z + dev_trans[14];
        });
    }).wait_and_throw();
    
    // copy results from device to host
    PointCloudXYZ transformed(N);
    q.memcpy(transformed[0].data(), dev_result, N * 4 * sizeof(PointType)).wait();
    
    sycl::free(dev_data, q);
    sycl::free(dev_trans, q);
    sycl::free(dev_result, q);
    
    return transformed;
}


int main() {
    
    /* Specity device */ 
    sycl::device dev; // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
    sycl::queue q(dev);
    
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

    // prepare data
    PointCloudXYZ points;
    points.emplace_back(1.0, 2.0, 3.0, 1.0);
    points.emplace_back(4.0, 5.0, 6.0, 1.0);
    points.emplace_back(7.0, 8.0, 9.0, 1.0);
    
    Eigen::Matrix4<PointType> trans = Eigen::Matrix4<PointType>::Identity();
    trans.block(0, 0, 3, 3) = Eigen::AngleAxis<PointType>(0.5 * M_PI, Eigen::Vector3<PointType>(0, 1, 0)).matrix(); // rotate 90 deg, y axis
    trans.block(0, 3, 3, 1) << 1.0, 2.0, 3.0;
    
    PointCloudXYZ transformed = transform_sycl(q, points, trans);
    
    // print result
    std::cout << "Src:\n";
    for (const auto& vec : points) {
        std::cout << vec.block(0, 0, 3, 1).transpose() << "\n"; 
    }
    std::cout << "transform matrix:\n";
    std::cout << trans << "\n";
    
    std::cout << "Result:\n";
    for (const auto& vec : transformed) {
        std::cout << vec.block(0, 0, 3, 1).transpose() << "\n"; 
    }
    // [4 4 2], [7 7 -1], [10 10 -4]
    return 0;
}

