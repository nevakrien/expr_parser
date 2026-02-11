#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define CHECK(x) if ((x) < 0) { perror("perf"); }

static int fd_l1_load   = -1;
static int fd_l1_miss   = -1;
static int fd_llc_load  = -1;
static int fd_llc_miss  = -1;
static int fd_cycles    = -1;
static int fd_instr     = -1;
static int fd_branches  = -1;
static int fd_br_miss   = -1;

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

    CHECK(fd_l1_load);
    CHECK(fd_l1_miss);
    CHECK(fd_llc_load);
    CHECK(fd_llc_miss);
    CHECK(fd_cycles);
    CHECK(fd_instr);
    CHECK(fd_branches);
    CHECK(fd_br_miss);
}

void perf_begin()
{
    int fds[] = {
        fd_l1_load, fd_l1_miss,
        fd_llc_load, fd_llc_miss,
        fd_cycles, fd_instr,
        fd_branches, fd_br_miss
    };

    for (int i = 0; i < 8; i++) {
        ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
        ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
    }
}

void perf_done(const char* name)
{
    uint64_t l1_load=0, l1_miss=0;
    uint64_t llc_load=0, llc_miss=0;
    uint64_t cycles=0, instr=0;
    uint64_t branches=0, br_miss=0;

    int fds[] = {
        fd_l1_load, fd_l1_miss,
        fd_llc_load, fd_llc_miss,
        fd_cycles, fd_instr,
        fd_branches, fd_br_miss
    };

    for (int i = 0; i < 8; i++)
        ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);

    read(fd_l1_load,  &l1_load,  8);
    read(fd_l1_miss,  &l1_miss,  8);
    read(fd_llc_load, &llc_load, 8);
    read(fd_llc_miss, &llc_miss, 8);
    read(fd_cycles,   &cycles,   8);
    read(fd_instr,    &instr,    8);
    read(fd_branches, &branches, 8);
    read(fd_br_miss,  &br_miss,  8);

    double l1_rate  = l1_load  ? (double)l1_miss  / l1_load  * 100.0 : 0.0;
    double llc_rate = llc_load ? (double)llc_miss / llc_load * 100.0 : 0.0;
    double cpi      = instr    ? (double)cycles   / instr    : 0.0;
    double ipc      = cycles   ? (double)instr    / cycles   : 0.0;
    double br_rate  = branches ? (double)br_miss  / branches * 100.0 : 0.0;

    printf("\n=== PERF REPORT: %s ===\n", name);
    printf("Instructions:      %llu\n", (unsigned long long)instr);
    printf("Cycles:            %llu\n", (unsigned long long)cycles);
    printf("IPC:               %.3f\n", ipc);
    printf("CPI:               %.3f\n", cpi);

    printf("\nBranches:          %llu\n", (unsigned long long)branches);
    printf("Branch misses:     %llu\n", (unsigned long long)br_miss);
    printf("Branch miss rate:  %.2f%%\n", br_rate);

    printf("\nL1 loads:          %llu\n", (unsigned long long)l1_load);
    printf("L1 misses:         %llu\n", (unsigned long long)l1_miss);
    printf("L1 miss rate:      %.2f%%\n", l1_rate);

    printf("\nLLC loads:         %llu\n", (unsigned long long)llc_load);
    printf("LLC misses:        %llu\n", (unsigned long long)llc_miss);
    printf("LLC miss rate:     %.2f%%\n", llc_rate);

    printf("=============================\n\n");
}
