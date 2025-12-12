#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
using namespace std;

static inline uint64_t mix64(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t calc_result(uint64_t iter, uint64_t job_idx) {
    uint64_t x = mix64(iter ^ job_idx);
    uint32_t spins = static_cast<uint32_t>(x & 0xFFU);
    for (uint32_t i = 0; i < spins; ++i) {
        x = mix64(x);
    }
    return x;
}

static inline void stress_kernel(uint64_t *dst, uint64_t iter, uint64_t job_idx) {
    array<uint64_t, 128> scratch;
    for (size_t i = 0; i < scratch.size(); ++i) {
        scratch[i] = iter + job_idx + i;
    }
    *dst = calc_result(iter, job_idx);
}

int main() {
    constexpr int64_t CAPACITY = 15;
    constexpr int64_t ITERATIONS = 5000;

    vector<uint64_t> output(CAPACITY, 0);

    int64_t max_dispatch_ns = 0;
    int64_t max_join_ns = 0;

    auto bench_start = chrono::steady_clock::now();
    for (int64_t iter_i = 0; iter_i < ITERATIONS; ++iter_i) {
        int64_t jobs = CAPACITY;
        if (iter_i % 5 == 1) {
            jobs = CAPACITY / 2;
        } else if (iter_i % 5 == 2) {
            jobs = 1;
        } else if (iter_i % 5 == 3) {
            jobs = (CAPACITY * 3) / 4;
        }

        vector<jthread> threads;
        threads.reserve(static_cast<size_t>(jobs));

        auto t0 = chrono::steady_clock::now();
        for (int64_t j = 0; j < jobs; ++j) {
            auto *dst = &output[static_cast<size_t>(j)];
            uint64_t iter = static_cast<uint64_t>(iter_i);
            uint64_t job = static_cast<uint64_t>(j);
            threads.emplace_back([dst, iter, job] {
                stress_kernel(dst, iter, job);
            });
        }
        auto t1 = chrono::steady_clock::now();
        for (auto &t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        auto t2 = chrono::steady_clock::now();

        auto dispatch_ns = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();
        auto join_ns = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
        if (dispatch_ns > max_dispatch_ns) {
            max_dispatch_ns = static_cast<int64_t>(dispatch_ns);
        }
        if (join_ns > max_join_ns) {
            max_join_ns = static_cast<int64_t>(join_ns);
        }

        for (int64_t j = 0; j < jobs; ++j) {
            uint64_t got = output[static_cast<size_t>(j)];
            uint64_t exp = calc_result(static_cast<uint64_t>(iter_i), static_cast<uint64_t>(j));
            if (got != exp) {
                cerr << "Mismatch at iter " << iter_i << " job " << j
                     << " got " << got << " expected " << exp << "\n";
                return 1;
            }
        }

        if (iter_i % 1000 == 0 && iter_i != 0) {
            cout << "ok through iter " << iter_i << "\n";
        }
    }
    auto bench_end = chrono::steady_clock::now();
    auto total_ns = chrono::duration_cast<chrono::nanoseconds>(bench_end - bench_start).count();

    cout << "Stress test passed.\n";
    cout << "max dispatch ns: " << max_dispatch_ns << "\n";
    cout << "max join ns: " << max_join_ns << "\n";
    cout << "total benchmark ns: " << total_ns << "\n";
    cout << "total benchmark: " << (total_ns / 1'000'000'000) << " s "
         << ((total_ns % 1'000'000'000) / 1'000'000) << " ms\n";
    return 0;
}

