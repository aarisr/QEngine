    // verify every shard of every card BEFORE first use, one thread per rank; a 2.3 GB object costs one core's crc time
    // otherwise; false on any mismatch (the caller treats it as a cache miss). Thread-creation failure falls back to
    // serial verification rather than failing a request.
    bool verify_all() const {
        if (d_.ng <= 0 || d_.ng > kMaxCards) return false;
        bool ok[kMaxCards] = {};
        std::vector<std::thread> th;
        try {
            th.reserve((size_t)d_.ng);
            for (int r = 0; r < d_.ng; r++)
                th.emplace_back([this, r, &ok] { ok[r] = verify_kv(r) && verify_st(r); });
        } catch (...) {
            for (auto& t : th) if (t.joinable()) t.join();
            for (int r = 0; r < d_.ng; r++) ok[r] = verify_kv(r) && verify_st(r);
        }
        for (auto& t : th) if (t.joinable()) t.join();
        for (int r = 0; r < d_.ng; r++) if (!ok[r]) return false;
        return true;
    }
