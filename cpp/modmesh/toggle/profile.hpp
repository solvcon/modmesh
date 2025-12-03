#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/base.hpp>
#include <chrono>

namespace modmesh
{

/**
 * Simple timer for wall time using high-resolution clock.
 */
class StopWatch
{

private:

    using clock_type = std::chrono::high_resolution_clock;
    using time_type = std::chrono::time_point<clock_type>;

public:

    /// A singleton.
    static StopWatch & me()
    {
        static StopWatch instance;
        return instance;
    }

    StopWatch()
        : m_start(clock_type::now())
        , m_stop(m_start)
    {
    }

    StopWatch(StopWatch const &) = default;
    StopWatch(StopWatch &&) = default;
    StopWatch & operator=(StopWatch const &) = default;
    StopWatch & operator=(StopWatch &&) = default;
    ~StopWatch() = default;

    /**
     * Return seconds between laps.
     */
    double lap()
    {
        m_start = m_stop;
        m_stop = clock_type::now();
        return std::chrono::duration<double>(m_stop - m_start).count();
    }

    /**
     * Return seconds between end and start.
     */
    double duration() const { return std::chrono::duration<double>(m_stop - m_start).count(); }

    /**
     * Return resolution in second.
     */
    static constexpr double resolution()
    {
        return double(clock_type::period::num) / double(clock_type::period::den);
    }

private:

    time_type m_start;
    time_type m_stop;

}; /* end struct StopWatch */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
