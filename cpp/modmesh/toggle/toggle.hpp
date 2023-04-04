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
#include <unordered_map>

#define MM_TOGGLE_CONSTEXPR_BOOL(NAME, VALUE) \
    static constexpr bool NAME = VALUE

namespace modmesh
{

int setenv(const char * name, const char * value, int overwrite);

struct DynamicToggleIndex
{

    enum Type : uint8_t
    {
        TYPE_NONE, // 0
        TYPE_BOOL,
        TYPE_INT8,
        TYPE_INT16,
        TYPE_INT32,
        TYPE_INT64,
        TYPE_REAL,
        TYPE_STRING
    };

    operator bool() const { return type != TYPE_NONE; }
    bool is_bool() const { return type == TYPE_BOOL; }
    bool is_int8() const { return type == TYPE_INT8; }
    bool is_int16() const { return type == TYPE_INT16; }
    bool is_int32() const { return type == TYPE_INT32; }
    bool is_int64() const { return type == TYPE_INT64; }
    bool is_real() const { return type == TYPE_REAL; }
    bool is_string() const { return type == TYPE_STRING; }

    // Index upper bound 2**32 is more than sufficient.  2**16 (65536) may be
    // too little.
    uint32_t index = 0;
    Type type = TYPE_NONE;

}; /* end struct DynamicToggleIndex */

class DynamicToggleTable
{

public:

    using keymap_type = std::unordered_map<std::string, DynamicToggleIndex>;

    static std::string const sentinel_string;

    bool get_bool(std::string const & key) const;
    void set_bool(std::string const & key, bool value);
    int8_t get_int8(std::string const & key) const;
    void set_int8(std::string const & key, int8_t value);
    int16_t get_int16(std::string const & key) const;
    void set_int16(std::string const & key, int16_t value);
    int32_t get_int32(std::string const & key) const;
    void set_int32(std::string const & key, int32_t value);
    int64_t get_int64(std::string const & key) const;
    void set_int64(std::string const & key, int64_t value);
    double get_real(std::string const & key) const;
    void set_real(std::string const & key, double value);
    std::string const & get_string(std::string const & key) const;
    void set_string(std::string const & key, std::string const & value);

    DynamicToggleIndex get_index(std::string const & key) const
    {
        auto it = m_key2index.find(key);
        return (it != m_key2index.end()) ? it->second : DynamicToggleIndex{0, DynamicToggleIndex::TYPE_NONE};
    }
    std::vector<std::string> keys() const;
    void clear();

private:

    keymap_type m_key2index;
    std::vector<bool> m_vector_bool;
    std::vector<int8_t> m_vector_int8;
    std::vector<int16_t> m_vector_int16;
    std::vector<int32_t> m_vector_int32;
    std::vector<int64_t> m_vector_int64;
    std::vector<double> m_vector_real;
    std::vector<std::string> m_vector_string;

}; /* end class DynamicToggleTable */

class Toggle
{

public:

#ifdef MODMESH_USE_PYSIDE
    MM_TOGGLE_CONSTEXPR_BOOL(USE_PYSIDE, true);
#else
    MM_TOGGLE_CONSTEXPR_BOOL(USE_PYSIDE, false);
#endif
    static Toggle & instance();

    Toggle(Toggle const &) = delete;
    Toggle(Toggle &&) = delete;
    Toggle & operator=(Toggle const &) = delete;
    Toggle & operator=(Toggle &&) = delete;
    ~Toggle() = default;

    bool get_show_axis() const { return m_show_axis; }
    void set_show_axis(bool v) { m_show_axis = v; }

    std::vector<std::string> dynamic_keys() const { return m_dynamic_table.keys(); }
    void dynamic_clear() { m_dynamic_table.clear(); }
    DynamicToggleIndex get_dynamic_index(std::string const & key) const { return m_dynamic_table.get_index(key); }

    bool get_bool(std::string const & key) const { return m_dynamic_table.get_bool(key); }
    void set_bool(std::string const & key, bool value) { m_dynamic_table.set_bool(key, value); }
    int8_t get_int8(std::string const & key) const { return m_dynamic_table.get_int8(key); }
    void set_int8(std::string const & key, int8_t value) { m_dynamic_table.set_int8(key, value); }
    int16_t get_int16(std::string const & key) const { return m_dynamic_table.get_int16(key); }
    void set_int16(std::string const & key, int16_t value) { m_dynamic_table.set_int16(key, value); }
    int32_t get_int32(std::string const & key) const { return m_dynamic_table.get_int32(key); }
    void set_int32(std::string const & key, int32_t value) { m_dynamic_table.set_int32(key, value); }
    int64_t get_int64(std::string const & key) const { return m_dynamic_table.get_int64(key); }
    void set_int64(std::string const & key, int64_t value) { m_dynamic_table.set_int64(key, value); }
    double get_real(std::string const & key) const { return m_dynamic_table.get_real(key); }
    void set_real(std::string const & key, double value) { m_dynamic_table.set_real(key, value); }
    std::string const & get_string(std::string const & key) const { return m_dynamic_table.get_string(key); }
    void set_string(std::string const & key, std::string const & value) { m_dynamic_table.set_string(key, value); }

private:

    Toggle() = default;

    bool m_show_axis = false;
    DynamicToggleTable m_dynamic_table;

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

    ProcessInfo & set_environment_variables();

    CommandLineInfo const & command_line() const { return m_command_line; }
    CommandLineInfo & command_line() { return m_command_line; }

private:

    ProcessInfo();

    CommandLineInfo m_command_line;

}; /* end class ProcessInfo */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
