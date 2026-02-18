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



#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#define CHECK(x) if ((x) < 0) { perror("perf"); }

/*
    HARD-CODED INTEL CORE PMU EVENT
    --------------------------------

    idq_uops_not_delivered.core

    Verified on this system via:
        /sys/bus/event_source/devices/cpu_core/format/

    Layout assumed:

        event  -> bits 0-7
        umask  -> bits 8-15
        inv    -> bit 23
        cmask  -> bits 24-31

    If this machine changes CPU generation, this may break.
    This is intentional — debugging tool only.
*/

/* Intel event encoding */
#define IDQ_EVENT  0x9c
#define IDQ_UMASK  0x01
#define IDQ_CMASK  0x04
#define IDQ_INV    1

static int fd_l1_load   = -1;
static int fd_l1_miss   = -1;
static int fd_llc_load  = -1;
static int fd_llc_miss  = -1;
static int fd_cycles    = -1;
static int fd_instr     = -1;
static int fd_branches  = -1;
static int fd_br_miss   = -1;
static int fd_frontend  = -1;

/* ------------------------------------------------ */

static int open_hw_cache(uint64_t config)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.type = PERF_TYPE_HW_CACHE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
}

static int open_hw(uint64_t config)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
}

/* open raw Intel core PMU event */
static int open_frontend_event(void)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);

    uint64_t config = 0;
    config |= (uint64_t)IDQ_EVENT;
    config |= (uint64_t)IDQ_UMASK << 8;
    config |= (uint64_t)IDQ_INV   << 23;
    config |= (uint64_t)IDQ_CMASK << 24;

    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
}

/* ------------------------------------------------ */

void perf_init()
{
    uint64_t l1_load =
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

    uint64_t l1_miss =
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    uint64_t llc_load =
        (PERF_COUNT_HW_CACHE_LL) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

    uint64_t llc_miss =
        (PERF_COUNT_HW_CACHE_LL) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    fd_l1_load  = open_hw_cache(l1_load);
    fd_l1_miss  = open_hw_cache(l1_miss);
    fd_llc_load = open_hw_cache(llc_load);
    fd_llc_miss = open_hw_cache(llc_miss);

    fd_cycles = open_hw(PERF_COUNT_HW_CPU_CYCLES);
    fd_instr  = open_hw(PERF_COUNT_HW_INSTRUCTIONS);

    fd_branches = open_hw(PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    fd_br_miss  = open_hw(PERF_COUNT_HW_BRANCH_MISSES);

    /* frontend stall counter */
    fd_frontend = open_frontend_event();

    CHECK(fd_l1_load);
    CHECK(fd_l1_miss);
    CHECK(fd_llc_load);
    CHECK(fd_llc_miss);
    CHECK(fd_cycles);
    CHECK(fd_instr);
    CHECK(fd_branches);
    CHECK(fd_br_miss);

    if (fd_frontend < 0)
        fprintf(stderr, "WARNING: frontend counter not available on this CPU\n");
}

/* ------------------------------------------------ */

void perf_begin()
{
    int fds[] = {
        fd_l1_load, fd_l1_miss,
        fd_llc_load, fd_llc_miss,
        fd_cycles, fd_instr,
        fd_branches, fd_br_miss,
        fd_frontend
    };

    for (int i = 0; i < 9; i++) {
        if (fds[i] < 0) continue;
        ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
        ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
    }
}

/* ------------------------------------------------ */

void perf_done(const char* name)
{
    uint64_t l1_load=0, l1_miss=0;
    uint64_t llc_load=0, llc_miss=0;
    uint64_t cycles=0, instr=0;
    uint64_t branches=0, br_miss=0;
    uint64_t frontend=0;

    int fds[] = {
        fd_l1_load, fd_l1_miss,
        fd_llc_load, fd_llc_miss,
        fd_cycles, fd_instr,
        fd_branches, fd_br_miss,
        fd_frontend
    };

    for (int i = 0; i < 9; i++)
        if (fds[i] >= 0)
            ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);

    read(fd_l1_load,  &l1_load,  8);
    read(fd_l1_miss,  &l1_miss,  8);
    read(fd_llc_load, &llc_load, 8);
    read(fd_llc_miss, &llc_miss, 8);
    read(fd_cycles,   &cycles,   8);
    read(fd_instr,    &instr,    8);
    read(fd_branches, &branches, 8);
    read(fd_br_miss,  &br_miss,  8);
    if (fd_frontend >= 0)
        read(fd_frontend, &frontend, 8);

    double l1_rate  = l1_load  ? (double)l1_miss  / l1_load  * 100.0 : 0.0;
    double llc_rate = llc_load ? (double)llc_miss / llc_load * 100.0 : 0.0;
    double ipc      = cycles   ? (double)instr    / cycles   : 0.0;
    double cpi      = instr    ? (double)cycles   / instr    : 0.0;
    double br_rate  = branches ? (double)br_miss  / branches * 100.0 : 0.0;

    double llc_of_total = l1_load ? (double)llc_load / l1_load * 100.0 : 0.0;
    double ram_of_total = l1_load ? (double)llc_miss / l1_load * 100.0 : 0.0;

    double frontend_pct =
        cycles ? (double)frontend / (4.0 * (double)cycles) * 100.0 : 0.0;

    printf("\n=== PERF REPORT: %s ===\n", name);
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

    if (fd_frontend >= 0) {
        printf("\nIDQ uops not delivered: %llu\n",
            (unsigned long long)frontend);
        printf("Frontend starvation:    %.2f%%\n", frontend_pct);
    }

    printf("=============================\n\n");
}
