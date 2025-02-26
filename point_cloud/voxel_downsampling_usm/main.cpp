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


inline auto voxel_downsampling_sycl(sycl::queue q, const PointCloudXYZ& points, double voxel_size) {
    // Ref: https://github.com/koide3/gtsam_points/blob/master/src/gtsam_points/types/point_cloud_cpu_funcs.cpp
    // voxelgrid_sampling
    // MIT License

    const size_t N = points.size();
    const PointType inv_voxel_size = 1.0 / voxel_size;

    constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
    constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3=63bits in 64bit int)
    constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
    constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive

    // compute bit on device
    std::vector<uint64_t> bits(N);
    {
        // allocate device memory
        auto* dev_data = sycl::malloc_device<PointType>(N * 4, q);
        auto* dev_inv_voxel_size = sycl::malloc_device<PointType>(1, q);
        auto* dev_bits = sycl::malloc_device<uint64_t>(N, q);
        q.memcpy(dev_data, points[0].data(), N * 4 * sizeof(PointType)).wait();
        q.memcpy(dev_inv_voxel_size, &inv_voxel_size, sizeof(PointType)).wait();
        q.fill(dev_bits, 0, N);

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                const auto inv_vs = *dev_inv_voxel_size;

                if (!sycl::isfinite(dev_data[i * 4 + 0]) || !sycl::isfinite(dev_data[i * 4 + 1]) || !sycl::isfinite(dev_data[i * 4 + 2])) {
                    dev_bits[i] = invalid_coord;
                    return;
                }
                const auto coord0 = static_cast<int64_t>(sycl::floor(dev_data[i * 4 + 0] * inv_vs)) + coord_offset;
                const auto coord1 = static_cast<int64_t>(sycl::floor(dev_data[i * 4 + 1] * inv_vs)) + coord_offset;
                const auto coord2 = static_cast<int64_t>(sycl::floor(dev_data[i * 4 + 2] * inv_vs)) + coord_offset;
                if (coord0 < 0 || coord_bit_mask < coord0 || coord1 < 0 || coord_bit_mask < coord1 || coord2 < 0 || coord_bit_mask < coord2) {
                    dev_bits[i] = invalid_coord;
                    return;
                }
                // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
                dev_bits[i] = (static_cast<uint64_t>(coord0 & coord_bit_mask) << (coord_bit_size * 0)) |
                              (static_cast<uint64_t>(coord1 & coord_bit_mask) << (coord_bit_size * 1)) |
                              (static_cast<uint64_t>(coord2 & coord_bit_mask) << (coord_bit_size * 2));
            });
        }).wait_and_throw();

        // copy results from device to host
        q.memcpy(bits.data(), dev_bits, N * sizeof(int64_t)).wait();

        // free device memory
        sycl::free(dev_bits, q);
        sycl::free(dev_data, q);
        sycl::free(dev_inv_voxel_size, q);
    }

    PointCloudXYZ result;
    {
        // prepare indices
        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);

        // sort indices by bits
        std::sort(indices.begin(), indices.end(),
            [&bits](size_t a, size_t b) { return bits[a] < bits[b]; });

        // compute average coords with same bit
        size_t counter = 1;
        auto& current_bit = bits[indices[0]];
        if (current_bit != invalid_coord) {
            result.emplace_back(points[indices[0]]);
            for(size_t i = 1; i < N; ++i) {
                if (bits[indices[i]] == invalid_coord) break;

                if (current_bit == bits[indices[i]]) {
                    result.back() += points[indices[i]];
                    ++counter;
                    if (i == N - 1) {
                        result.back() /= counter;
                    }
                } else {
                    result.back() /= counter;
                    counter = 1;
                    current_bit = bits[indices[i]];
                    result.emplace_back(points[indices[i]]);
                }
            }
        }
    }
    return result;
}


int main() {

    /* Specity device */
    sycl::device dev; // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
    sycl::queue q(dev);

    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

    // prepare data
    PointCloudXYZ points;
    points.emplace_back(1.0, 1.0, 3.0, 1.0);
    points.emplace_back(1.0, 2.0, 3.0, 1.0);
    points.emplace_back(1.0, 3.0, 3.0, 1.0);
    points.emplace_back(4.0, 5.0, 6.0, 1.0);
    points.emplace_back(4.0, 5.5, 6.0, 1.0);
    points.emplace_back(7.0, 8.0, 9.0, 1.0);

    auto transformed = voxel_downsampling_sycl(q, points, 2.0);

    // print result
    std::cout << "Src:\n";
    for (const auto& vec : points) {
        std::cout << vec.block(0, 0, 3, 1).transpose() << "\n";
    }
    std::cout << "Result:\n";
    for (const auto& vec : transformed) {
        std::cout << vec.block(0, 0, 3, 1).transpose() << "\n";
    }
    return 0;
}

