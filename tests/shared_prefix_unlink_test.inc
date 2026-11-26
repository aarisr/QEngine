static bool concurrent_unlink_test(const std::string& path, const std::string& fp) {
    std::atomic<bool> ready{false}, stop{false}, valid{true};
    std::thread reader([&] {
        Map mapped;
        if (open_map(path, fp, mapped) != Rd::OK) { valid = false; ready = true; return; }
        ready = true;
        while (!stop.load()) {
            if (!mapped.verify_all()) { valid = false; break; }
        }
    });
    for (int i = 0; i < 1000 && !ready.load(); i++) usleep(1000);
    const bool opened = ready.load() && valid.load();
    const int rc = unlink(path.c_str());
    stop = true;
    reader.join();
    return opened && rc == 0 && valid.load();
}
