#pragma once
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <iostream>

#define PROFILE(cmd) telef::solver::profile::profile([&](){ cmd })
#define PROFILE_DECLARE(s) std::chrono::time_point<std::chrono::system_clock> timeStart_##s, timeEnd_##s
#define PROFILE_START(s) timeStart_##s = std::chrono::high_resolution_clock::now()
#define PROFILE_END(s) timeEnd_##s = std::chrono::high_resolution_clock::now()
#define PROFILE_GET(s) std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd_##s - timeStart_##s).count()
#define PROFILE_GET_CUSTOM(s, res) std::chrono::duration_cast<res>(timeStart_##s - timeEnd_##s).count()


namespace telef::solver::profile {
    template <typename Duration = std::chrono::nanoseconds,
              typename F,
              typename ... Args>
    typename Duration::rep profile(F&& fun,  Args&&... args) {
        const auto t_begin = std::chrono::high_resolution_clock::now();
        std::forward<F>(fun)(std::forward<Args>(args)...);
        const auto t_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<Duration>(t_end - t_begin).count();
    }

    template <typename Duration = std::chrono::nanoseconds,
              typename F>
    typename Duration::rep profile(F f) {
        const auto t_begin = std::chrono::high_resolution_clock::now();
        f();
        const auto t_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<Duration>(t_end - t_begin).count();
    }
}