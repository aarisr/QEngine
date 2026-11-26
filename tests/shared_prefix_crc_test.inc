static void test_crc_performance(int& failures) {
    const char* known = "123456789";
    EXPECT(crc32c_soft(known, 9) == 0xe3069283u);
    EXPECT(crc32c_hw(known, 9) == 0xe3069283u);
    EXPECT(crc32c(known, 9) == 0xe3069283u);
    if (!crc32c_hardware()) {
        printf("MEASURE crc32c hw=unavailable; software fallback active\n");
        return;
    }
    std::vector<char> data(32u << 20);
    uint32_t seed = 17;
    for (auto& c : data) { seed = seed * 1664525u + 1013904223u; c = (char)(seed >> 24); }
    std::vector<double> hw_ms, sw_ms;
    volatile uint32_t sink = 0;
    for (int i = 0; i < 11; i++) {
        double t = now_s();
        sink = crc32c_hw(data.data(), data.size());
        hw_ms.push_back((now_s() - t) * 1000);
        t = now_s();
        const uint32_t sw = crc32c_soft(data.data(), data.size());
        sw_ms.push_back((now_s() - t) * 1000);
        EXPECT(sink == sw);
    }
    const double mib = data.size() / (1024.0 * 1024.0);
    printf("MEASURE crc32c_hardware GB/s n=%zu p50=%.2f p95=%.2f\n", hw_ms.size(),
           mib / (q(hw_ms, 0.5) / 1000) / 1024.0, mib / (q(hw_ms, 0.95) / 1000) / 1024.0);
    printf("MEASURE crc32c_software GB/s n=%zu p50=%.2f p95=%.2f\n", sw_ms.size(),
           mib / (q(sw_ms, 0.5) / 1000) / 1024.0, mib / (q(sw_ms, 0.95) / 1000) / 1024.0);
    (void)sink;
    (void)failures;
}
