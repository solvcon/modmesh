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
#include <modmesh/toggle/RadixTree.hpp>

#include <string>
#include <vector>
#include <unordered_map>

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
        TYPE_STRING,
        TYPE_SUBKEY
    };

    operator bool() const { return type != TYPE_NONE; }
    bool is_bool() const { return type == TYPE_BOOL; }
    bool is_int8() const { return type == TYPE_INT8; }
    bool is_int16() const { return type == TYPE_INT16; }
    bool is_int32() const { return type == TYPE_INT32; }
    bool is_int64() const { return type == TYPE_INT64; }
    bool is_real() const { return type == TYPE_REAL; }
    bool is_string() const { return type == TYPE_STRING; }
    bool is_subkey() const { return type == TYPE_SUBKEY; }

    // Index upper bound 2**32 is more than sufficient.  2**16 (65536) may be
    // too little.
    uint32_t index = 0;
    Type type = TYPE_NONE;

}; /* end struct DynamicToggleIndex */

class DynamicToggleTable;

class HierarchicalToggleAccess
{

public:

    explicit HierarchicalToggleAccess(DynamicToggleTable & table)
        : m_table(&table)
    {
    }

    HierarchicalToggleAccess(DynamicToggleTable & table, std::string base)
        : m_table(&table)
        , m_base(std::move(base))
    {
    }

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

    HierarchicalToggleAccess get_subkey(std::string const & key);
    void add_subkey(std::string const & key);

    DynamicToggleIndex get_index(std::string const & key) const;

    std::string rekey(std::string const & key) const
    {
        return m_base.empty() ? key : (Formatter() << m_base << "." << key);
    }

private:

    DynamicToggleTable * m_table = nullptr;
    std::string m_base;

}; /* end class HierarchicalToggleAccess */

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

    HierarchicalToggleAccess get_subkey(std::string const & key)
    {
        return HierarchicalToggleAccess(*this, key);
    }
    void add_subkey(std::string const & key);

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

inline DynamicToggleIndex HierarchicalToggleAccess::get_index(std::string const & key) const
{
    return m_table->get_index(rekey(key));
}

#define MM_TOGGLE_SOLID_BOOL(NAME)         \
public:                                    \
    bool NAME() const { return m_##NAME; } \
                                           \
private:                                   \
    bool m_##NAME;

class SolidToggle
{

public:

    SolidToggle();

    MM_TOGGLE_SOLID_BOOL(use_pyside)

}; /* end class SolidToggle */

#define MM_TOGGLE_FIXED_BOOL(NAME, DEFAULT)      \
public:                                          \
    bool get_##NAME() const { return m_##NAME; } \
    void set_##NAME(bool v) { m_##NAME = v; }    \
                                                 \
private:                                         \
    bool m_##NAME = DEFAULT;

class FixedToggle
{

public:

    MM_TOGGLE_FIXED_BOOL(python_redirect, true)
    MM_TOGGLE_FIXED_BOOL(show_axis, false)

}; /* end class FixedToggle */

/**
 * The toggle system for modmesh. There are 3 types of toggles:
 *
 * 1. solid toggles: managed by SolidToggle class. It is the toggles whose value
 *    is determined during compile time. The value is read-only (const) through
 *    out the program lifecycle (the process).
 *
 *    The solid toggles have address and can be referenced. They cannot be
 *    optimized out (unlike macros and constexpr). It could add overhead when
 *    used in tight loops. The overhead may usually be too low to be noticed,
 *    but sometimes one needs to be careful about it.
 *
 * 2. fixed toggles: managed by FixedToggle class. It is the toggles whose name
 *    is determined during compile time. The value can be changed during
 *    runtime.
 *
 *    Because the names are determined during compile time, when accessing the
 *    toggles, no table lookup is needed. The address of the toggle variables
 *    has been determined by the compiler and linker.
 *
 *    The runtime cost of fixed toggles is the same as solid toggles. It may
 *    be used in tight loops. Just becareful about the potential runtime
 *    overhead.
 *
 * 3. dynamic toggles: managed by DynamicToggleTable. The toggles are
 *    hierarchical and the names and values can be added, removed, and modified
 *    during runtime. The value needs to use limited data types: bool, int8,
 *    int16, int32, int64, real, and string. It is intentional not to support
 *    unsigned integers.
 *
 *    Accessing dynamic toggles requires table lookup and string comparison. It
 *    is slow but flexible.
 *
 *    To access the dynamic toggles from C++, the data type of the toggle
 *    The hierarchical access (from C++) uses ".", like:
 *
 *      tg.get_int8("top_level.second_level.key_name")
 *
 *    In Python, the wrapper can determine the type dynamically, and the
 *    hierarchical access may use attribute syntax:
 *
 *      tg.top_level.second_level.key_name = value
 */
class Toggle
{

public:

    static Toggle & instance();

    Toggle * clone() const { return new Toggle(*this); }

    ~Toggle() = default;

    SolidToggle const & solid() const { return m_solid; }

    FixedToggle const & fixed() const { return m_fixed; }
    FixedToggle & fixed() { return m_fixed; }

    DynamicToggleTable const & dynamic() const { return m_dynamic_table; }
    DynamicToggleTable & dynamic() { return m_dynamic_table; }

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
    HierarchicalToggleAccess get_subkey(std::string const & key) { return m_dynamic_table.get_subkey(key); }
    void add_subkey(std::string const & key) { m_dynamic_table.add_subkey(key); }

private:

    Toggle() = default;
    Toggle(Toggle const &) = default;
    Toggle(Toggle &&) = default;
    Toggle & operator=(Toggle const &) = default;
    Toggle & operator=(Toggle &&) = default;

    SolidToggle m_solid;
    FixedToggle m_fixed;
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
