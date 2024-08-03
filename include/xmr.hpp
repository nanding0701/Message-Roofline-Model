#pragma once

#include <algorithm>
#include <assert.h>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <execution>
#include <functional>
#include <iostream>
#include <random>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <sched.h>
#include <span>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <getopt.h>

// timer class
class Timer {
   public:
    Timer() { start(); }

    ~Timer() { stop(); }

    void start() { start_time_point = std::chrono::high_resolution_clock::now(); }

    double stop() {
        end_time_point = std::chrono::high_resolution_clock::now();
        return duration();
    }

    double duration() {
        auto start =
            std::chrono::time_point_cast<std::chrono::microseconds>(start_time_point).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time_point).time_since_epoch().count();
        auto duration = end - start;
        double ms = duration * 1e-6;
        return ms;
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_point;
};

// function to generate a big random value
int64_t bigRandVal() {

  int64_t random =
    (((int64_t) rand() <<  0) & 0x000000000000FFFFull) |
    (((int64_t) rand() << 16) & 0x00000000FFFF0000ull) |
    (((int64_t) rand() << 32) & 0x0000FFFF00000000ull) |
    (((int64_t) rand() << 48) & 0x0FFF000000000000ull);
  return random;

}

constexpr int KB = 1024;
constexpr int MB = 1024 * KB;
constexpr int GB = 1024 * MB;
