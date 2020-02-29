#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD-style license; see COPYING
 */

#include "modmesh/base.hpp"

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

    /// A global singleton.
    static StopWatch & me()
    {
        static StopWatch instance;
        return instance;
    }

    StopWatch() { lap(); }

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
        m_start = m_end;
        m_end = clock_type::now();
        return std::chrono::duration<double>(m_end - m_start).count();
    }

    /**
     * Return resolution in second.
     */
    static constexpr double resolution()
    {
        return double(clock_type::period::num) / double(clock_type::period::den);
    }

private:

    time_type m_start;
    time_type m_end;

}; /* end struct StopWatch */

struct TimedEntry
{
    size_t m_count = 0;
    double m_time = 0.0;
}; /* end struct TimedEntry */

class TimeRegistry
{

public:

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
                << "count = " << it->second.m_count << " , "
                << "time = " << it->second.m_time << " (second)"
                << std::endl;
        }
        return ostm.str();
    }

    void add(const char * name, double time)
    {
        auto it = m_entry.find(name);
        if (it == m_entry.end())
        {
            it = std::get<0>(m_entry.insert({name, {0, 0.0}}));
        }
        ++(it->second.m_count);
        it->second.m_time += time;
    }

    ~TimeRegistry()
    {
        // Uncomment for debugging.
        //std::cout << report();
    }

private:

    TimeRegistry() = default;
    TimeRegistry(TimeRegistry const & ) = delete;
    TimeRegistry(TimeRegistry       &&) = delete;
    TimeRegistry & operator=(TimeRegistry const & ) = delete;
    TimeRegistry & operator=(TimeRegistry       &&) = delete;

    std::map<const char *, TimedEntry> m_entry;

}; /* end struct TimeRegistry */

struct ScopedTimer
{

    ScopedTimer() = delete;

    ScopedTimer(const char * name) : m_name(name) {}

    ~ScopedTimer()
    {
        TimeRegistry::me().add(m_name, m_sw.lap());
    }

    StopWatch m_sw;
    char const * m_name;

}; /* end struct ScopedTimer */

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
