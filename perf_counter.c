// #include <sys/prctl.h>
// #include <linux/prctl.h>

// void perf_init(){}
// void perf_begin()
// {
//     // prctl(PR_TASK_PERF_EVENTS_ENABLE);
// }

// void perf_done()
// {
//     // prctl(PR_TASK_PERF_EVENTS_DISABLE);
// }

// #include <fcntl.h>
// #include <unistd.h>

// #include <fcntl.h>
// #include <unistd.h>
// #include <errno.h>
// #include <stdio.h>
// #include <sched.h>

// static int perf_fd = -1;

// void perf_init()
// {
//     // open without blocking program startup
//     while (1) {
//         perf_fd = open("/tmp/perfctl", O_WRONLY | O_NONBLOCK);
//         if (perf_fd >= 0)
//             break;

//         if (errno == ENXIO) {
//             // perf not ready yet
//             sched_yield();
//             continue;
//         }
//         perror("perf fifo open");
//         break;
//     }
// }

// static inline void perf_sync()
// {
//     // serialize CPU so region starts AFTER enable
//     asm volatile("" ::: "memory");
// }

// void perf_begin()
// {
//     if (perf_fd >= 0) {
//         write(perf_fd, "enable\n", 7);
//         perf_sync();
//     }
// }

// void perf_done()
// {
//     if (perf_fd >= 0) {
//         perf_sync();
//         write(perf_fd, "disable\n", 8);
//     }
// }

// #define _GNU_SOURCE
// #include <linux/perf_event.h>
// #include <sys/syscall.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <stdint.h>
// #include <string.h>
// #include <stdio.h>

// #define CHECK(x) if ((x) < 0) { perror("perf"); }

// static int fd_l1_load   = -1;
// static int fd_l1_miss   = -1;
// static int fd_llc_load  = -1;
// static int fd_llc_miss  = -1;
// static int fd_cycles    = -1;
// static int fd_instr     = -1;
// static int fd_branches  = -1;
// static int fd_br_miss   = -1;

// static int open_hw_cache(uint64_t config)
// {
//     struct perf_event_attr attr;
//     memset(&attr, 0, sizeof(attr));

//     attr.type = PERF_TYPE_HW_CACHE;
//     attr.size = sizeof(attr);
//     attr.config = config;
//     attr.disabled = 1;
//     attr.exclude_kernel = 1;
//     attr.exclude_hv = 1;

//     return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
// }

// static int open_hw(uint64_t config)
// {
//     struct perf_event_attr attr;
//     memset(&attr, 0, sizeof(attr));

//     attr.type = PERF_TYPE_HARDWARE;
//     attr.size = sizeof(attr);
//     attr.config = config;
//     attr.disabled = 1;
//     attr.exclude_kernel = 1;
//     attr.exclude_hv = 1;

//     return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
// }

// void perf_init()
// {
//     uint64_t l1_load =
//         (PERF_COUNT_HW_CACHE_L1D) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

//     uint64_t l1_miss =
//         (PERF_COUNT_HW_CACHE_L1D) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

//     uint64_t llc_load =
//         (PERF_COUNT_HW_CACHE_LL) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

//     uint64_t llc_miss =
//         (PERF_COUNT_HW_CACHE_LL) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

//     fd_l1_load  = open_hw_cache(l1_load);
//     fd_l1_miss  = open_hw_cache(l1_miss);
//     fd_llc_load = open_hw_cache(llc_load);
//     fd_llc_miss = open_hw_cache(llc_miss);

//     fd_cycles = open_hw(PERF_COUNT_HW_CPU_CYCLES);
//     fd_instr  = open_hw(PERF_COUNT_HW_INSTRUCTIONS);

//     fd_branches = open_hw(PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
//     fd_br_miss  = open_hw(PERF_COUNT_HW_BRANCH_MISSES);

//     CHECK(fd_l1_load);
//     CHECK(fd_l1_miss);
//     CHECK(fd_llc_load);
//     CHECK(fd_llc_miss);
//     CHECK(fd_cycles);
//     CHECK(fd_instr);
//     CHECK(fd_branches);
//     CHECK(fd_br_miss);
// }

// void perf_begin()
// {
//     int fds[] = {
//         fd_l1_load, fd_l1_miss,
//         fd_llc_load, fd_llc_miss,
//         fd_cycles, fd_instr,
//         fd_branches, fd_br_miss
//     };

//     for (int i = 0; i < 8; i++) {
//         ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
//         ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
//     }
// }

// void perf_done(const char* name)
// {
//     uint64_t l1_load=0, l1_miss=0;
//     uint64_t llc_load=0, llc_miss=0;
//     uint64_t cycles=0, instr=0;
//     uint64_t branches=0, br_miss=0;

//     int fds[] = {
//         fd_l1_load, fd_l1_miss,
//         fd_llc_load, fd_llc_miss,
//         fd_cycles, fd_instr,
//         fd_branches, fd_br_miss
//     };

//     for (int i = 0; i < 8; i++)
//         ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);

//     read(fd_l1_load,  &l1_load,  8);
//     read(fd_l1_miss,  &l1_miss,  8);
//     read(fd_llc_load, &llc_load, 8);
//     read(fd_llc_miss, &llc_miss, 8);
//     read(fd_cycles,   &cycles,   8);
//     read(fd_instr,    &instr,    8);
//     read(fd_branches, &branches, 8);
//     read(fd_br_miss,  &br_miss,  8);

//     double l1_rate  = l1_load  ? (double)l1_miss  / l1_load  * 100.0 : 0.0;
//     double llc_rate = llc_load ? (double)llc_miss / llc_load * 100.0 : 0.0;
//     double cpi      = instr    ? (double)cycles   / instr    : 0.0;
//     double ipc      = cycles   ? (double)instr    / cycles   : 0.0;
//     double br_rate  = branches ? (double)br_miss  / branches * 100.0 : 0.0;

//     /* new derived metrics */
//     double llc_of_total = l1_load ? (double)llc_load / l1_load * 100.0 : 0.0;
//     double ram_of_total = l1_load ? (double)llc_miss / l1_load * 100.0 : 0.0;

//     printf("\n=== PERF REPORT: %s ===\n", name);
//     printf("Instructions:      %llu\n", (unsigned long long)instr);
//     printf("Cycles:            %llu\n", (unsigned long long)cycles);
//     printf("IPC:               %.3f\n", ipc);
//     printf("CPI:               %.3f\n", cpi);

//     printf("\nBranches:          %llu\n", (unsigned long long)branches);
//     printf("Branch misses:     %llu\n", (unsigned long long)br_miss);
//     printf("Branch miss rate:  %.2f%%\n", br_rate);

//     printf("\nL1 loads:          %llu\n", (unsigned long long)l1_load);
//     printf("L1 misses:         %llu\n", (unsigned long long)l1_miss);
//     printf("L1 miss rate:      %.2f%%\n", l1_rate);

//     printf("\nLLC loads:         %llu\n", (unsigned long long)llc_load);
//     printf("LLC misses:        %llu\n", (unsigned long long)llc_miss);
//     printf("LLC miss rate:     %.2f%%  (given you reached LLC)\n", llc_rate);

//     printf("\nLLC accesses / total loads: %.4f%%\n", llc_of_total);
//     printf("RAM accesses / total loads: %.4f%%\n", ram_of_total);

//     printf("=============================\n\n");

// }



// #define _GNU_SOURCE
// #include <linux/perf_event.h>
// #include <sys/syscall.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <stdint.h>
// #include <string.h>
// #include <stdio.h>
// #include <errno.h>

// #define CHECK(x) if ((x) < 0) { perror("perf"); }

// /*
//     HARD-CODED INTEL CORE PMU EVENT
//     --------------------------------

//     idq_uops_not_delivered.core

//     Verified on this system via:
//         /sys/bus/event_source/devices/cpu_core/format/

//     Layout assumed:

//         event  -> bits 0-7
//         umask  -> bits 8-15
//         inv    -> bit 23
//         cmask  -> bits 24-31

//     If this machine changes CPU generation, this may break.
//     This is intentional — debugging tool only.
// */

// /* Intel event encoding */
// #define IDQ_EVENT  0x9c
// #define IDQ_UMASK  0x01
// #define IDQ_CMASK  0x04
// #define IDQ_INV    1

// static int fd_l1_load   = -1;
// static int fd_l1_miss   = -1;
// static int fd_llc_load  = -1;
// static int fd_llc_miss  = -1;
// static int fd_cycles    = -1;
// static int fd_instr     = -1;
// static int fd_branches  = -1;
// static int fd_br_miss   = -1;
// static int fd_frontend  = -1;

// /* ------------------------------------------------ */

// static int open_hw_cache(uint64_t config)
// {
//     struct perf_event_attr attr;
//     memset(&attr, 0, sizeof(attr));

//     attr.type = PERF_TYPE_HW_CACHE;
//     attr.size = sizeof(attr);
//     attr.config = config;
//     attr.disabled = 1;
//     attr.exclude_kernel = 1;
//     attr.exclude_hv = 1;

//     return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
// }

// static int open_hw(uint64_t config)
// {
//     struct perf_event_attr attr;
//     memset(&attr, 0, sizeof(attr));

//     attr.type = PERF_TYPE_HARDWARE;
//     attr.size = sizeof(attr);
//     attr.config = config;
//     attr.disabled = 1;
//     attr.exclude_kernel = 1;
//     attr.exclude_hv = 1;

//     return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
// }

// /* open raw Intel core PMU event */
// static int open_frontend_event(void)
// {
//     struct perf_event_attr attr;
//     memset(&attr, 0, sizeof(attr));

//     attr.type = PERF_TYPE_RAW;
//     attr.size = sizeof(attr);

//     uint64_t config = 0;
//     config |= (uint64_t)IDQ_EVENT;
//     config |= (uint64_t)IDQ_UMASK << 8;
//     config |= (uint64_t)IDQ_INV   << 23;
//     config |= (uint64_t)IDQ_CMASK << 24;

//     attr.config = config;
//     attr.disabled = 1;
//     attr.exclude_kernel = 1;
//     attr.exclude_hv = 1;

//     return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
// }

// /* ------------------------------------------------ */

// void perf_init()
// {
//     uint64_t l1_load =
//         (PERF_COUNT_HW_CACHE_L1D) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

//     uint64_t l1_miss =
//         (PERF_COUNT_HW_CACHE_L1D) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

//     uint64_t llc_load =
//         (PERF_COUNT_HW_CACHE_LL) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

//     uint64_t llc_miss =
//         (PERF_COUNT_HW_CACHE_LL) |
//         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
//         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

//     fd_l1_load  = open_hw_cache(l1_load);
//     fd_l1_miss  = open_hw_cache(l1_miss);
//     fd_llc_load = open_hw_cache(llc_load);
//     fd_llc_miss = open_hw_cache(llc_miss);

//     fd_cycles = open_hw(PERF_COUNT_HW_CPU_CYCLES);
//     fd_instr  = open_hw(PERF_COUNT_HW_INSTRUCTIONS);

//     fd_branches = open_hw(PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
//     fd_br_miss  = open_hw(PERF_COUNT_HW_BRANCH_MISSES);

//     /* frontend stall counter */
//     fd_frontend = open_frontend_event();

//     CHECK(fd_l1_load);
//     CHECK(fd_l1_miss);
//     CHECK(fd_llc_load);
//     CHECK(fd_llc_miss);
//     CHECK(fd_cycles);
//     CHECK(fd_instr);
//     CHECK(fd_branches);
//     CHECK(fd_br_miss);

//     if (fd_frontend < 0)
//         fprintf(stderr, "WARNING: frontend counter not available on this CPU\n");
// }

// /* ------------------------------------------------ */

// void perf_begin()
// {
//     int fds[] = {
//         fd_l1_load, fd_l1_miss,
//         fd_llc_load, fd_llc_miss,
//         fd_cycles, fd_instr,
//         fd_branches, fd_br_miss,
//         fd_frontend
//     };

//     for (int i = 0; i < 9; i++) {
//         if (fds[i] < 0) continue;
//         ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
//         ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
//     }
// }

// /* ------------------------------------------------ */

// void perf_done(const char* name)
// {
//     uint64_t l1_load=0, l1_miss=0;
//     uint64_t llc_load=0, llc_miss=0;
//     uint64_t cycles=0, instr=0;
//     uint64_t branches=0, br_miss=0;
//     uint64_t frontend=0;

//     int fds[] = {
//         fd_l1_load, fd_l1_miss,
//         fd_llc_load, fd_llc_miss,
//         fd_cycles, fd_instr,
//         fd_branches, fd_br_miss,
//         fd_frontend
//     };

//     for (int i = 0; i < 9; i++)
//         if (fds[i] >= 0)
//             ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);

//     read(fd_l1_load,  &l1_load,  8);
//     read(fd_l1_miss,  &l1_miss,  8);
//     read(fd_llc_load, &llc_load, 8);
//     read(fd_llc_miss, &llc_miss, 8);
//     read(fd_cycles,   &cycles,   8);
//     read(fd_instr,    &instr,    8);
//     read(fd_branches, &branches, 8);
//     read(fd_br_miss,  &br_miss,  8);
//     if (fd_frontend >= 0)
//         read(fd_frontend, &frontend, 8);

//     double l1_rate  = l1_load  ? (double)l1_miss  / l1_load  * 100.0 : 0.0;
//     double llc_rate = llc_load ? (double)llc_miss / llc_load * 100.0 : 0.0;
//     double ipc      = cycles   ? (double)instr    / cycles   : 0.0;
//     double cpi      = instr    ? (double)cycles   / instr    : 0.0;
//     double br_rate  = branches ? (double)br_miss  / branches * 100.0 : 0.0;

//     double llc_of_total = l1_load ? (double)llc_load / l1_load * 100.0 : 0.0;
//     double ram_of_total = l1_load ? (double)llc_miss / l1_load * 100.0 : 0.0;

//     double frontend_pct =
//         cycles ? (double)frontend / (4.0 * (double)cycles) * 100.0 : 0.0;

//     printf("\n=== PERF REPORT: %s ===\n", name);
//     printf("Instructions:      %llu\n", (unsigned long long)instr);
//     printf("Cycles:            %llu\n", (unsigned long long)cycles);
//     printf("IPC:               %.3f\n", ipc);
//     printf("CPI:               %.3f\n", cpi);

//     printf("\nBranches:          %llu\n", (unsigned long long)branches);
//     printf("Branch misses:     %llu\n", (unsigned long long)br_miss);
//     printf("Branch miss rate:  %.2f%%\n", br_rate);

//     printf("\nL1 miss rate:      %.2f%%\n", l1_rate);
//     printf("LLC miss rate:     %.2f%%  (given you reached LLC)\n", llc_rate);
//     printf("LLC accesses/all:  %.4f%%\n", llc_of_total);
//     printf("RAM accesses/all:  %.4f%%\n", ram_of_total);

//     if (fd_frontend >= 0) {
//         printf("\nIDQ uops not delivered: %llu\n",
//             (unsigned long long)frontend);
//         printf("Frontend starvation:    %.2f%%\n", frontend_pct);
//     }

//     printf("=============================\n\n");
// }

#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

/*
    HARD-CODED INTEL CORE PMU EVENT
    --------------------------------
    idq_uops_not_delivered.core

    Layout assumed:
        event  -> bits 0-7
        umask  -> bits 8-15
        inv    -> bit 23
        cmask  -> bits 24-31
*/

#define IDQ_EVENT  0x9c
#define IDQ_UMASK  0x01
#define IDQ_CMASK  0x04
#define IDQ_INV    1

/* ------------------------ policy knobs ------------------------ */

#ifndef PERF_HARD_FAIL
#define PERF_HARD_FAIL 0   /* 0 = warn + continue, 1 = exit(1) on failure */
#endif

static void perf_fail(const char* what) {
    int e = errno;
    fprintf(stderr, "perf: %s: %s (errno=%d)\n", what, strerror(e), e);
    if (PERF_HARD_FAIL) exit(1);
}

/* Full read helper: avoids ignored-return-value warnings and catches short reads. */
static uint64_t read_u64_full(int fd, const char* name) {
    uint64_t v = 0;
    size_t off = 0;
    unsigned char* p = (unsigned char*)&v;

    while (off < sizeof(v)) {
        ssize_t n = read(fd, p + off, sizeof(v) - off);
        if (n == 0) {
            errno = EIO;
            fprintf(stderr, "perf: short read (EOF) for %s\n", name);
            if (PERF_HARD_FAIL) exit(1);
            return 0;
        }
        if (n < 0) {
            fprintf(stderr, "perf: read failed for %s: %s\n", name, strerror(errno));
            if (PERF_HARD_FAIL) exit(1);
            return 0;
        }
        off += (size_t)n;
    }
    return v;
}

/* ------------------------ event table ------------------------ */

typedef enum {
    EVT_HW,
    EVT_HW_CACHE,
    EVT_RAW,
} evt_kind_t;

typedef struct {
    const char* name;
    evt_kind_t  kind;
    uint64_t    config;
    int         optional;   /* if 1: missing is allowed */
    int         fd;
} perf_counter_t;

/* Cache config helper: constant expression version. */
#define CACHE_CFG(cache, op, res) \
    ((uint64_t)(cache) | ((uint64_t)(op) << 8) | ((uint64_t)(res) << 16))

/* Raw Intel IDQ event config: constant expression version. */
#define IDQ_RAW_CFG() ( \
    ((uint64_t)IDQ_EVENT) | \
    ((uint64_t)IDQ_UMASK << 8) | \
    ((uint64_t)IDQ_INV   << 23) | \
    ((uint64_t)IDQ_CMASK << 24) \
)

static perf_counter_t g_ctrs[] = {
    { "l1_load",   EVT_HW_CACHE, CACHE_CFG(PERF_COUNT_HW_CACHE_L1D,
                                           PERF_COUNT_HW_CACHE_OP_READ,
                                           PERF_COUNT_HW_CACHE_RESULT_ACCESS), 0, -1 },
    { "l1_miss",   EVT_HW_CACHE, CACHE_CFG(PERF_COUNT_HW_CACHE_L1D,
                                           PERF_COUNT_HW_CACHE_OP_READ,
                                           PERF_COUNT_HW_CACHE_RESULT_MISS),   0, -1 },
    { "llc_load",  EVT_HW_CACHE, CACHE_CFG(PERF_COUNT_HW_CACHE_LL,
                                           PERF_COUNT_HW_CACHE_OP_READ,
                                           PERF_COUNT_HW_CACHE_RESULT_ACCESS), 0, -1 },
    { "llc_miss",  EVT_HW_CACHE, CACHE_CFG(PERF_COUNT_HW_CACHE_LL,
                                           PERF_COUNT_HW_CACHE_OP_READ,
                                           PERF_COUNT_HW_CACHE_RESULT_MISS),   0, -1 },

    { "cycles",    EVT_HW,       (uint64_t)PERF_COUNT_HW_CPU_CYCLES,         0, -1 },
    { "instr",     EVT_HW,       (uint64_t)PERF_COUNT_HW_INSTRUCTIONS,       0, -1 },
    { "branches",  EVT_HW,       (uint64_t)PERF_COUNT_HW_BRANCH_INSTRUCTIONS,0, -1 },
    { "br_miss",   EVT_HW,       (uint64_t)PERF_COUNT_HW_BRANCH_MISSES,      0, -1 },

    { "idq_uops_not_delivered.core", EVT_RAW, IDQ_RAW_CFG(),                 1, -1 },
};


static const int g_nctrs = (int)(sizeof(g_ctrs) / sizeof(g_ctrs[0]));

/* Group leader: make enable/disable/reset atomic-ish and consistent. */
static int g_group_leader_fd = -1;

/* ------------------------ open helpers ------------------------ */

static int open_one_counter(perf_counter_t* c, int group_fd) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);

    switch (c->kind) {
        case EVT_HW:      attr.type = PERF_TYPE_HARDWARE; break;
        case EVT_HW_CACHE:attr.type = PERF_TYPE_HW_CACHE; break;
        case EVT_RAW:     attr.type = PERF_TYPE_RAW;      break;
        default:          errno = EINVAL; return -1;
    }

    attr.config = c->config;
    attr.disabled = 1;

    /* Typical for userland microbench / debugging. */
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    /* You can add:
       attr.inherit = 1;  // if you want child threads/processes
       attr.pinned = 1;   // if you want to force scheduling (may fail more often)
    */

    int fd = (int)syscall(__NR_perf_event_open, &attr, 0, -1, group_fd, 0);
    return fd;
}

/* ------------------------ public API ------------------------ */

void perf_init(void) {
    /* Open leader first: pick something that should exist. */
    g_group_leader_fd = open_one_counter(&g_ctrs[0], -1);
    if (g_group_leader_fd < 0) {
        perf_fail("perf_event_open (group leader)");
        /* If we can't open a leader, we can't really do anything. */
        if (!PERF_HARD_FAIL) return;
    }
    g_ctrs[0].fd = g_group_leader_fd;

    /* Open the rest in the group. */
    for (int i = 1; i < g_nctrs; i++) {
        int fd = open_one_counter(&g_ctrs[i], g_group_leader_fd);
        if (fd < 0) {
            if (g_ctrs[i].optional) {
                fprintf(stderr, "perf: optional counter unavailable: %s (%s)\n",
                        g_ctrs[i].name, strerror(errno));
                g_ctrs[i].fd = -1;
                continue;
            }
            char buf[256];
            snprintf(buf, sizeof(buf), "perf_event_open (%s)", g_ctrs[i].name);
            perf_fail(buf);
            g_ctrs[i].fd = -1;
            continue;
        }
        g_ctrs[i].fd = fd;
    }
}

void perf_begin(void) {
    if (g_group_leader_fd < 0) return;

    /* Reset entire group, then enable group. */
    if (ioctl(g_group_leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0)
        perf_fail("ioctl RESET (group)");
    if (ioctl(g_group_leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
        perf_fail("ioctl ENABLE (group)");
}

void perf_done(const char* name) {
    if (g_group_leader_fd < 0) return;

    if (ioctl(g_group_leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0)
        perf_fail("ioctl DISABLE (group)");

    /* Pull raw counts. */
    uint64_t l1_load   = (g_ctrs[0].fd >= 0) ? read_u64_full(g_ctrs[0].fd, g_ctrs[0].name) : 0;
    uint64_t l1_miss   = (g_ctrs[1].fd >= 0) ? read_u64_full(g_ctrs[1].fd, g_ctrs[1].name) : 0;
    uint64_t llc_load  = (g_ctrs[2].fd >= 0) ? read_u64_full(g_ctrs[2].fd, g_ctrs[2].name) : 0;
    uint64_t llc_miss  = (g_ctrs[3].fd >= 0) ? read_u64_full(g_ctrs[3].fd, g_ctrs[3].name) : 0;
    uint64_t cycles    = (g_ctrs[4].fd >= 0) ? read_u64_full(g_ctrs[4].fd, g_ctrs[4].name) : 0;
    uint64_t instr     = (g_ctrs[5].fd >= 0) ? read_u64_full(g_ctrs[5].fd, g_ctrs[5].name) : 0;
    uint64_t branches  = (g_ctrs[6].fd >= 0) ? read_u64_full(g_ctrs[6].fd, g_ctrs[6].name) : 0;
    uint64_t br_miss   = (g_ctrs[7].fd >= 0) ? read_u64_full(g_ctrs[7].fd, g_ctrs[7].name) : 0;

    uint64_t frontend  = 0;
    if (g_ctrs[8].fd >= 0)
        frontend = read_u64_full(g_ctrs[8].fd, g_ctrs[8].name);

    /* Derived metrics. */
    double l1_rate  = l1_load  ? (double)l1_miss  / (double)l1_load  * 100.0 : 0.0;
    double llc_rate = llc_load ? (double)llc_miss / (double)llc_load * 100.0 : 0.0;
    double ipc      = cycles   ? (double)instr    / (double)cycles   : 0.0;
    double cpi      = instr    ? (double)cycles   / (double)instr    : 0.0;
    double br_rate  = branches ? (double)br_miss  / (double)branches * 100.0 : 0.0;

    double llc_of_total = l1_load ? (double)llc_load / (double)l1_load * 100.0 : 0.0;
    double ram_of_total = l1_load ? (double)llc_miss / (double)l1_load * 100.0 : 0.0;

    double frontend_pct =
        cycles ? (double)frontend / (4.0 * (double)cycles) * 100.0 : 0.0;

    printf("\n=== PERF REPORT: %s ===\n", name ? name : "(null)");
    printf("Instructions:      %llu\n", (unsigned long long)instr);
    printf("Cycles:            %llu\n", (unsigned long long)cycles);
    printf("IPC:               %.3f\n", ipc);
    printf("CPI:               %.3f\n", cpi);

    printf("\nBranches:          %llu\n", (unsigned long long)branches);
    printf("Branch misses:     %llu\n", (unsigned long long)br_miss);
    printf("Branch miss rate:  %.2f%%\n", br_rate);

    printf("\nL1 miss rate:      %.2f%%\n", l1_rate);
    printf("LLC miss rate:     %.2f%%  (given you reached LLC)\n", llc_rate);
    printf("LLC accesses/all:  %.4f%%\n", llc_of_total);
    printf("RAM accesses/all:  %.4f%%\n", ram_of_total);

    if (g_ctrs[8].fd >= 0) {
        printf("\nIDQ uops not delivered: %llu\n", (unsigned long long)frontend);
        printf("Frontend starvation:    %.2f%%\n", frontend_pct);
    }

    printf("=============================\n\n");
}

// #define _GNU_SOURCE
// #include <linux/perf_event.h>
// #include <sys/syscall.h>
// #include <sys/ioctl.h>
// #include <unistd.h>
// #include <stdint.h>
// #include <string.h>
// #include <stdio.h>
// #include <errno.h>
// #include <stdlib.h>

// #define DIE(...) do { \
//     fprintf(stderr, "perf: "); \
//     fprintf(stderr, __VA_ARGS__); \
//     fprintf(stderr, " (%s, errno=%d)\n", strerror(errno), errno); \
//     exit(1); \
// } while (0)

// static int fd_group = -1;   /* branches (leader) */
// static int fd_miss  = -1;   /* branch-misses (member) */

// static int perf_open(struct perf_event_attr* attr, pid_t pid, int cpu, int group_fd) {
//     return (int)syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, 0);
// }

// static void setup_attr(struct perf_event_attr* a, uint64_t config) {
//     memset(a, 0, sizeof(*a));
//     a->type = PERF_TYPE_HARDWARE;
//     a->size = sizeof(*a);
//     a->config = config;

//     a->disabled = 1;
//     a->exclude_kernel = 1;
//     a->exclude_hv = 1;

//     /* Read as a group + include timing so we can scale if multiplexed. */
//     a->read_format =
//         PERF_FORMAT_GROUP |
//         PERF_FORMAT_TOTAL_TIME_ENABLED |
//         PERF_FORMAT_TOTAL_TIME_RUNNING;
// }

// /* read group for exactly 2 counters:
//    u64 nr
//    u64 time_enabled
//    u64 time_running
//    u64 value0
//    u64 value1
// */
// typedef struct {
//     uint64_t nr;
//     uint64_t time_enabled;
//     uint64_t time_running;
//     uint64_t values[2];
// } read_group2_t;

// static read_group2_t read_group2_full(int fd) {
//     read_group2_t rg;
//     memset(&rg, 0, sizeof(rg));

//     size_t off = 0;
//     unsigned char* p = (unsigned char*)&rg;
//     while (off < sizeof(rg)) {
//         ssize_t n = read(fd, p + off, sizeof(rg) - off);
//         if (n < 0) DIE("read(group) failed");
//         if (n == 0) { errno = EIO; DIE("read(group) EOF"); }
//         off += (size_t)n;
//     }
//     return rg;
// }

// static double scale_count(uint64_t raw, uint64_t enabled, uint64_t running) {
//     if (running == 0) return 0.0;
//     if (running == enabled) return (double)raw;
//     return (double)raw * (double)enabled / (double)running;
// }

// /* ------------------ public API (same names) ------------------ */

// void perf_init(void) {
//     pid_t pid = 0;  /* current thread */
//     int cpu = -1;   /* any CPU */

//     struct perf_event_attr a_br, a_miss;
//     setup_attr(&a_br,   PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
//     setup_attr(&a_miss, PERF_COUNT_HW_BRANCH_MISSES);

//     fd_group = perf_open(&a_br, pid, cpu, -1);
//     if (fd_group < 0) DIE("perf_event_open(branches) failed");

//     fd_miss = perf_open(&a_miss, pid, cpu, fd_group);
//     if (fd_miss < 0) DIE("perf_event_open(branch-misses) failed");
// }

// void perf_begin(void) {
//     if (fd_group < 0) return;

//     if (ioctl(fd_group, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0)
//         DIE("ioctl RESET(group) failed");
//     if (ioctl(fd_group, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
//         DIE("ioctl ENABLE(group) failed");
// }

// void perf_done(const char* name) {
//     if (fd_group < 0) return;

//     if (ioctl(fd_group, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0)
//         DIE("ioctl DISABLE(group) failed");

//     read_group2_t rg = read_group2_full(fd_group);
//     if (rg.nr < 2) {
//         errno = EINVAL;
//         DIE("expected 2 counters in group, got %llu", (unsigned long long)rg.nr);
//     }

//     double branches = scale_count(rg.values[0], rg.time_enabled, rg.time_running);
//     double misses   = scale_count(rg.values[1], rg.time_enabled, rg.time_running);
//     double rate     = (branches > 0.0) ? (misses / branches * 100.0) : 0.0;

//     printf("\n=== PERF (branch-only): %s ===\n", name ? name : "(null)");
//     printf("time_enabled:  %llu\n", (unsigned long long)rg.time_enabled);
//     printf("time_running:  %llu\n", (unsigned long long)rg.time_running);
//     if (rg.time_running != rg.time_enabled)
//         printf("NOTE: multiplexed; counts scaled by enabled/running\n");

//     printf("Branches:      %.0f\n", branches);
//     printf("Branch misses: %.0f\n", misses);
//     printf("Miss rate:     %.2f%%\n", rate);
//     printf("================================\n\n");
// }
