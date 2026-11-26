if (const char* pages = env_str("QE_SHARED_L2_PAGES")) {
    if (!strcmp(pages, "thp")) sp_pages_advice = MADV_HUGEPAGE;
    else if (!strcmp(pages, "4k")) sp_pages_advice = MADV_NOHUGEPAGE;
}
