#include <iostream>
#include <dlfcn.h>
#include "nccl/nccl_net_v8.h" // Ensure this header file exists and includes the definition of ncclNet_v8_t


ncclNet_v8_t* ncclNets[1] = { nullptr};
extern "C" ncclNet_v8_t* ncclNetPlugin_v8();
int main() {
    void* handle = dlopen("target/release/libnccl_net_jnpr.so", RTLD_LAZY);
    if (!handle) {
        std::cout << "Failed to load library\n";
        return 1;
    }

    std::cout << "opened handle" << std::endl;

    /*
    ncclNets[0] = (ncclNet_v8_t*)dlsym(handle, "ncclNetPlugin_v8");
        if (ncclNets[0] == nullptr) {
        std::cout << "Failed to find symbol\n";
    }
    std::cout << "name" << ncclNets[0]->name << std::endl;
    */
    ncclNet_v8_t *(*get_nccl_net)() = (ncclNet_v8_t* (*)())dlsym(handle, "ncclNetPlugin_v8");
    if (!get_nccl_net) {
        std::cout << "Failed to find symbol\n";
        return 1;
    }
    std::cout << "found symbol" << std::endl;

    ncclNets[0] = get_nccl_net();
    if (!ncclNets[0]) {
        std::cerr << "Failed to initialize ncclNet_v8\n";
        dlclose(handle);
        return 1;
    }

    std::cout << "Loaded plugin: " << ncclNets[0]->name << std::endl;

    ncclNets[0]->init(NULL);

    int ndev;
    ncclNets[0]->devices(&ndev);
    std::cout << "Number of devices: " << ndev << std::endl;


    void* handle 


    dlclose(handle);

    std::cout << "Done\n";

    return 0;
}