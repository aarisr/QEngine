// CPU unit test of src/idle_poll.h: the idle VRAM poll runs rarely, work and a shrunk pool keep the full cadence.
#include "idle_poll.h"
#include <cstdio>
#include <cstdlib>
#define EXPECT(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)
int main() {
    using qe_idle_poll::due;
    double last = 0;
    EXPECT(due(true, false, 100.0, &last, 30));    // first idle call polls
    EXPECT(!due(true, false, 101.0, &last, 30));
    EXPECT(!due(true, false, 129.0, &last, 30));
    EXPECT(due(true, false, 130.0, &last, 30));    // 30 s later
    EXPECT(!due(true, false, 131.0, &last, 30));
    EXPECT(due(false, false, 131.5, &last, 30));   // work: always
    EXPECT(due(true, true, 131.6, &last, 30));     // shrunk pool: always
    EXPECT(last == 130.0);
    EXPECT(due(true, false, 131.7, &last, 0));     // 0: the old behaviour
    EXPECT(due(true, false, 131.8, &last, 0));
    puts("idle_poll_unit ok");
    return 0;
}
