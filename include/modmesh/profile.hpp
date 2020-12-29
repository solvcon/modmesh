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

#include <string>
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

    StopWatch() : m_start(clock_type::now()), m_stop(m_start) {}

    StopWatch(StopWatch const & ) = default;
    StopWatch(StopWatch       &&) = default;
    StopWatch & operator=(StopWatch const & ) = default;
    StopWatch & operator=(StopWatch       &&) = default;
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

class TimedEntry
{

public:

    size_t count() const { return m_count; }
    double time() const { return m_time; }

    double start() { return m_sw.lap(); }
    double stop()
    {
        double const time = m_sw.lap();
        add_time(time);
        return time;
    }

    TimedEntry & add_time(double time)
    {
        ++m_count;
        m_time += time;
        return *this;
    }

private:

    size_t m_count = 0;
    double m_time = 0.0;
    StopWatch m_sw;

}; /* end class TimedEntry */

class TimeRegistry
{

public:

    /// The singleton.
    static TimeRegistry & me()
    {
        static TimeRegistry inst;
        return inst;
    }

    std::string report() const
    {
        std::ostringstream ostm;
        for (auto it = m_entry.begin() ; it != m_entry.end() ; ++it)
        {
            ostm
                << it->first << " : "
                << "count = " << it->second.count() << " , "
                << "time = " << it->second.time() << " (second)"
                << std::endl;
        }
        return ostm.str();
    }

    void add(std::string const & name, double time)
    {
        entry(name).add_time(time);
    }

    void add(const char * name, double time)
    {
        add(std::string(name), time);
    }

    std::vector<std::string> names() const
    {
        std::vector<std::string> ret;
        for (auto const & item : m_entry)
        {
            ret.push_back(item.first);
        }
        return ret;
    }

    TimedEntry & entry(std::string const & name)
    {
        auto it = m_entry.find(name);
        if (it == m_entry.end())
        {
            it = std::get<0>(m_entry.insert({name, {}}));
        }
        return it->second;
    }

    void clear() { m_entry.clear(); }

    TimeRegistry(TimeRegistry const & ) = delete;
    TimeRegistry(TimeRegistry       &&) = delete;
    TimeRegistry & operator=(TimeRegistry const & ) = delete;
    TimeRegistry & operator=(TimeRegistry       &&) = delete;

    ~TimeRegistry() // NOLINT(modernize-use-equals-default)
    {
        // Uncomment for debugging.
        //std::cout << report();
    }

private:

    TimeRegistry() = default;

    std::map<std::string, TimedEntry> m_entry;

}; /* end struct TimeRegistry */

class ScopedTimer
{

public:

    ScopedTimer() = delete;
    ScopedTimer(ScopedTimer const & ) = delete;
    ScopedTimer(ScopedTimer       &&) = delete;
    ScopedTimer & operator=(ScopedTimer const & ) = delete;
    ScopedTimer & operator=(ScopedTimer       &&) = delete;

    explicit ScopedTimer(const char * name) : m_name(name) {}

    ~ScopedTimer()
    {
        TimeRegistry::me().add(m_name, m_sw.lap());
    }

private:

    StopWatch m_sw;
    char const * m_name;

}; /* end class ScopedTimer */

} /* end namespace modmesh */

/*
 * MODMESH_PROFILE defined: Enable profiling API.
 */
#ifdef MODMESH_PROFILE

#define MODMESH_TIME(NAME) \
    ScopedTimer _local_scoped_timer_ ## __LINE__(NAME);

/*
 * No MODMESH_PROFILE defined: Disable profiling API.
 */
#else // MODMESH_PROFILE

#define MODMESH_TIME(NAME)

#endif // MODMESH_PROFILE
/*
 * End MODMESH_PROFILE.
 */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
