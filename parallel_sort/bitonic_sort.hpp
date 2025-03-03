// SPDX-FileCopyrightText: Copyright 2024 fateshelled
// SPDX-License-Identifier: Apache-2.0

#ifndef BITONIC_SORT_HPP
#define BITONIC_SORT_HPP

#include <vector>
#include <cmath>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif


template <typename T>
void bitonic_sort(sycl::queue& q, const std::vector<T> &input, std::vector<T> &result) {
    const int N = input.size();

    int pow2_size = 1;
    while (pow2_size < N) {
        pow2_size *= 2;
    }

    T* d_data = sycl::malloc_device<T>(pow2_size, q);

    std::vector<T> padded_data(pow2_size, std::numeric_limits<T>::max());
    std::copy(input.begin(), input.end(), padded_data.begin());

    q.memcpy(d_data, padded_data.data(), pow2_size * sizeof(T)).wait();

    // work group size
    const int local_size = std::min(256, (int)q.get_device().get_info<sycl::info::device::max_work_group_size>());

    // sort
    for (int k = 2; k <= pow2_size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(pow2_size / 2), sycl::range<1>(local_size)),
                    [=](sycl::nd_item<1> item) {
                        int gid = item.get_global_id(0);

                        if (gid >= pow2_size / 2) return;

                        int i = 2 * gid - (gid & (j - 1));
                        int ixj = i ^ j;

                        if (ixj > i && i < pow2_size && ixj < pow2_size) {
                            const bool direction = (i & k) == 0;

                            if ((d_data[i] > d_data[ixj]) == direction) {
                                T temp = d_data[i];
                                d_data[i] = d_data[ixj];
                                d_data[ixj] = temp;
                            }
                        }
                    }
                );
            }).wait();
        }
    }

    result.resize(N);
    q.memcpy(result.data(), d_data, N * sizeof(T)).wait();
    sycl::free(d_data, q);
}

#endif // BITONIT_SORT_HPP
