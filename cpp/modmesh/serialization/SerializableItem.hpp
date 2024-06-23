#pragma once

/*
 * Copyright (c) 2024, An-Chi Liu <phy.tiger@gmail.com>
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

#include <iomanip>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <modmesh/base.hpp> // for helper macros

namespace modmesh
{

class SerializableItem
{
public:
    virtual std::string to_json() const = 0;
    virtual void from_json(const std::string & json) = 0;

    // TODO: Add more serialization methods, e.g., to/from binary, to/from YAML.
}; /* end class SerializableItem */

namespace detail
{

/// Escape special characters in a string.
std::string escape_string(std::string_view str_view);

template <typename T>
std::string to_json_string(const T & value)
{
    if constexpr (std::is_base_of_v<SerializableItem, T>)
    {
        return value.to_json(); /* recursive here */
    }
    else if constexpr (std::is_convertible_v<T, std::string>)
    {
        return "\"" + escape_string(value) + "\"";
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        return value ? "true" : "false";
    }
    else if constexpr (is_specialization_of<std::vector, T>::value)
    {
        std::ostringstream oss;
        oss << "[";
        const char * separator = "";
        for (const auto & item : value)
        {
            oss << separator << to_json_string(item); /* recursive here */
            separator = ",";
        }
        oss << "]";
        return oss.str();
    }
    else
    {
        return std::to_string(value);
    }
}

}; /* end namespace detail */

/// Serialize a class with member variables.
/// Use `register_member("key", class.member);` to add members when using this macro
/// Note that the order of members in the JSON string is based on the order of `register_member` calls.
#define MM_DECL_SERIALIZABLE(...)                                                       \
public:                                                                                 \
    std::string to_json() const override                                                \
    {                                                                                   \
        std::ostringstream oss;                                                         \
        oss << "{";                                                                     \
        const char * separator = "";                                                    \
        auto register_member = [&](const char * name, auto && value)                    \
        {                                                                               \
            oss << separator << "\"" << name << "\":" << detail::to_json_string(value); \
            separator = ",";                                                            \
        };                                                                              \
        __VA_ARGS__                                                                     \
        oss << "}";                                                                     \
        return oss.str();                                                               \
    }                                                                                   \
                                                                                        \
    void from_json(const std::string & json) override                                   \
    {                                                                                   \
        /* TODO: The implemenation is in the next PR */                                 \
    } /* end MM_DECL_SERIALIZABLE*/

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
