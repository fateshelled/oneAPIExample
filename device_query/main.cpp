#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

int main(int argc, char** argv) {

    for (auto platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()<< std::endl;
            std::cout << "\ttype: "
                      << (device.is_cpu() ? "CPU" : "GPU")<< std::endl;
            std::cout << "\tVendorID: "
                      << device.get_info<sycl::info::device::vendor_id>() << std::endl;
            std::cout << "\tBackend version: "
                      << device.get_info<sycl::info::device::backend_version>() << std::endl;
            std::cout << "\tDriver version: "
                    << device.get_info<sycl::info::device::driver_version>() << std::endl;
            std::cout << "\tGlobal Memory Size: "
                        << device.get_info<sycl::info::device::global_mem_size>() / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
            std::cout << "\tMax Clock Frequency: "
                    << device.get_info<sycl::info::device::max_clock_frequency>() / 1000.0 << " GHz" << std::endl;
            std::cout << "\tDouble precision support: "
                    << (device.has(sycl::aspect::fp64) ? "true": "false") << std::endl;
            std::cout << "\tAvailable: "
            << (device.get_info<sycl::info::device::is_available>() ? "true" : "false") << std::endl;
        }
    }

    return 0;
}
