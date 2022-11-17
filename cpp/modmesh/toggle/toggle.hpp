#pragma once

/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/toggle/profile.hpp>

#include <string>
#include <vector>

namespace modmesh
{

class Toggle
{

public:

    static Toggle & instance();

    Toggle(Toggle const &) = delete;
    Toggle(Toggle &&) = delete;
    Toggle & operator=(Toggle const &) = delete;
    Toggle & operator=(Toggle &&) = delete;
    ~Toggle() = default;

    bool get_show_axis() const { return m_show_axis; }
    void set_show_axis(bool v) { m_show_axis = v; }

private:

    Toggle() = default;

    bool m_show_axis;

}; /* end class Toggle */

class ProcessInfo;

class CommandLineInfo
{

public:

    CommandLineInfo() = default;
    CommandLineInfo(CommandLineInfo const &) = default;
    // NOLINTNEXTLINE(bugprone-exception-escape)
    CommandLineInfo(CommandLineInfo &&) = default;
    CommandLineInfo & operator=(CommandLineInfo const &) = default;
    CommandLineInfo & operator=(CommandLineInfo &&) = default;
    ~CommandLineInfo() = default;

    std::string const & executable_basename() const { return m_executable_basename; }
    std::vector<std::string> const & populated_argv() const { return m_populated_argv; }
    std::vector<std::string> const & python_argv() const { return m_python_argv; }
    void set_python_argv(std::vector<std::string> const & argv)
    {
        if (!m_frozen)
        {
            m_python_argv = argv;
        }
    }

    class PopulatePasskey
    {
        friend ProcessInfo;
    };

    void populate(int argc, char ** argv, PopulatePasskey const &)
    {
        populate(argc, argv, /* repopulate */ false);
    }

    void freeze() { m_frozen = true; }

    bool frozen() const { return m_frozen; }
    bool populated() const { return m_populated; }

    bool python_main() const { return m_python_main; }
    int python_main_argc() const { return m_python_main_argc; }
    char ** python_main_argv_ptr() { return m_python_main_argv_ptr.data(); }

private:

    void unfreeze() { m_frozen = false; }

    void populate(int argc, char ** argv, bool repopulate);

    bool m_frozen = false;
    bool m_populated = false;
    std::string m_executable_basename;
    std::vector<std::string> m_populated_argv;
    std::vector<std::string> m_python_argv;

    bool m_python_main = false;
    int m_python_main_argc = 0;
    SimpleArray<char> m_python_main_argv_char;
    SimpleArray<char *> m_python_main_argv_ptr;

}; /* end class CommandLineInfo */

// NOLINTNEXTLINE(bugprone-exception-escape)
class ProcessInfo
{

public:

    static ProcessInfo & instance();

    ProcessInfo & populate_command_line(int argc, char ** argv)
    {
        m_command_line.populate(argc, argv, CommandLineInfo::PopulatePasskey{});
        return *this;
    }

    CommandLineInfo const & command_line() const { return m_command_line; }
    CommandLineInfo & command_line() { return m_command_line; }

private:

    ProcessInfo();

    CommandLineInfo m_command_line;

}; /* end class ProcessInfo */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
