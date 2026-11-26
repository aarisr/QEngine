if (x == cudaSuccess) {
    bool readonly = !e->devs.empty();
    for (int dev : e->devs) {
        int supported = 0;
        if (cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterReadOnlySupported, dev) != cudaSuccess || !supported)
            readonly = false;
    }
    const unsigned flags = cudaHostRegisterPortable | (readonly ? cudaHostRegisterReadOnly : 0u);
    x = cudaHostRegister(p, n, flags);
    if (x != cudaSuccess && readonly) {
        (void)cudaGetLastError();
        x = cudaHostRegister(p, n, cudaHostRegisterPortable);
    }
}
