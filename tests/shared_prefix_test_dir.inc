static bool init_test_dir() {
    const char* root = getenv("QE_SP_TEST_DIR");
    const std::string base = root && *root ? root : "/tmp";
    std::string pattern = base + "/qengine-shared-prefix-unit.XXXXXX";
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    char* made = mkdtemp(buf.data());
    if (!made) return false;
    g_dir = made;
    return true;
}
static void cleanup_test_dir() {
    DIR* d = opendir(g_dir.c_str());
    if (d) {
        while (dirent* de = readdir(d)) {
            const std::string nm = de->d_name;
            if (nm == "." || nm == "..") continue;
            unlink((g_dir + "/" + nm).c_str());
        }
        closedir(d);
    }
    rmdir(g_dir.c_str());
}
