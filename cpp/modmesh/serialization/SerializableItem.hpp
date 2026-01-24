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

#include <algorithm>
#include <concepts>
#include <format>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
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

template <typename T>
concept JsonSerializableConcept = std::is_base_of_v<SerializableItem, T>;

template <typename T>
concept JsonStringConcept = std::is_convertible_v<T, std::string>;

template <typename T>
concept JsonBooleanConcept = std::is_same_v<T, bool>;

template <typename T>
concept JsonIntegralConcept = std::is_integral_v<T> && !JsonBooleanConcept<T>;

template <typename T>
concept JsonVectorConcept = is_specialization_of_v<std::vector, T>;

template <typename T>
concept JsonMapConcept = is_specialization_of_v<std::unordered_map, T> && std::is_same_v<typename T::key_type, std::string>;

std::string escape_string(std::string_view str_view);

/// Trim leading and trailing whitespaces and control characters.
std::string trim_string(const std::string & str);

/// State of the JSON parser.
enum class JsonState
{
    Start,
    ObjectKey,
    Column,
    Comma,
    ObjectValue,
    End,
}; /* end enum class JsonState */

/// Type of JSON token.
enum class JsonType
{
    Object,
    Array,
    String,
    Number,
    Boolean,
    Null,
    Unknown,
}; /* end enum class JsonType */

struct JsonNode; // Forward declaration
using JsonMap = std::unordered_map<std::string, std::unique_ptr<JsonNode>>;
using JsonArray = std::vector<std::unique_ptr<JsonNode>>;

struct JsonNode
{
    using JsonValue = std::variant<JsonMap, JsonArray, std::string>;

    JsonType type;
    JsonValue value;

    JsonNode(JsonType type, const std::string & expression)
        : type(type)
    {
        parse(expression);
    }

private:

    void parse(const std::string & expression)
    {
        if (type == JsonType::Object)
        {
            value = std::move(parse_object(expression));
        }
        else if (type == JsonType::Array)
        {
            value = std::move(parse_array(expression));
        }
        else
        {
            value = expression;
        }
    }

    static JsonMap parse_object(const std::string & json);

    static JsonArray parse_array(const std::string & json);

}; /* end struct JsonNode */

/// Helper class for JSON serialization and deserialization.
class JsonHelper
{
public:
    template <typename T>
    static std::string to_json_string(const T & value);

    template <typename T>
    // TODO: have a design that can remove the output argument to increase the maintainability
    static void from_json_string(const std::unique_ptr<JsonNode> & node, T & value);

    template <typename T>
    // TODO: have a design that can remove the output argument to increase the maintainability
    static void from_json_string(const std::unique_ptr<JsonNode> & node, std::vector<T> & vec);

}; /* end class JsonHelper */

template <typename T>
std::string JsonHelper::to_json_string(const T & value)
{
    if constexpr (JsonSerializableConcept<T>)
    {
        // NOLINTNEXTLINE(misc-no-recursion)
        return value.to_json(); /* recursive here */
    }
    else if constexpr (JsonStringConcept<T>)
    {
        return std::format("\"{}\"", escape_string(value));
    }
    else if constexpr (JsonBooleanConcept<T>)
    {
        return value ? "true" : "false";
    }
    else if constexpr (JsonVectorConcept<T>)
    {
        std::ostringstream oss;
        oss << "[";

        constexpr auto transform_function = [](const auto & item)
        {
            // NOLINTNEXTLINE(misc-no-recursion)
            return to_json_string(item); /* recursive here */
        };
        auto json_items = value | std::views::transform(transform_function);

        const char * separator = "";
        for (const auto & json_item : json_items)
        {
            oss << separator << json_item;
            separator = ",";
        }
        oss << "]";
        return oss.str();
    }
    else if constexpr (JsonMapConcept<T>)
    {
        auto keys = value | std::views::keys | std::ranges::to<std::vector>();
        std::ranges::sort(keys);

        std::ostringstream oss;
        oss << "{";
        const char * separator = "";
        for (const auto & key : keys)
        {
            // NOLINTNEXTLINE(misc-no-recursion)
            oss << separator
                << std::format("\"{}\":{}", key, to_json_string(value.at(key))); /* recursive here */
            separator = ",";
        }
        oss << "}";
        return oss.str();
    }
    else
    {
        return std::to_string(value);
    }
}

template <typename T>
void JsonHelper::from_json_string(const std::unique_ptr<JsonNode> & node, T & value)
{
    if (node->type == JsonType::Null)
    {
        return; /* TODO: properly handle null case */
    }

    if constexpr (JsonIntegralConcept<T>)
    {
        if (node->type != JsonType::Number)
        {
            throw std::runtime_error("Invalid JSON format: invalid integer type.");
        }
        value = std::stoll(std::get<std::string>(node->value));
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        if (node->type != JsonType::Number)
        {
            throw std::runtime_error("Invalid JSON format: invalid floating number type.");
        }
        value = std::stod(std::get<std::string>(node->value));
    }
    else if constexpr (JsonStringConcept<T>)
    {
        if (node->type != JsonType::String)
        {
            throw std::runtime_error("Invalid JSON format: invalid string type.");
        }
        const auto & str = std::get<std::string>(node->value);
        value = str.substr(1, str.size() - 2); // remove the surrounding quotes
    }
    else if constexpr (JsonBooleanConcept<T>)
    {
        if (node->type != JsonType::Boolean)
        {
            throw std::runtime_error("Invalid JSON format: invalid boolean type.");
        }
        value = std::get<std::string>(node->value) != "false";
    }
    else if constexpr (JsonSerializableConcept<T>)
    {
        if (node->type != JsonType::Object)
        {
            throw std::runtime_error("Invalid JSON format: invalid object type.");
        }
        // NOLINTNEXTLINE(misc-no-recursion)
        value.from_json(std::get<JsonMap>(node->value)); /* recursive here */
    }
    else if constexpr (JsonMapConcept<T>)
    {
        if (node->type != JsonType::Object)
        {
            throw std::runtime_error("Invalid JSON format: invalid object type.");
        }
        const auto & obj = std::get<JsonMap>(node->value);
        for (const auto & [key, json_node] : obj)
        {
            // NOLINTNEXTLINE(misc-no-recursion)
            from_json_string(json_node, value[key]); /* recursive here */
        }
    }
    else
    {
        throw std::runtime_error("Invalid JSON format: invalid type.");
    }
}

template <typename T>
void JsonHelper::from_json_string(const std::unique_ptr<JsonNode> & node, std::vector<T> & vec)
{
    if (node->type == JsonType::Null)
    {
        return; /* TODO: properly handle null case */
    }

    if (node->type != JsonType::Array)
    {
        throw std::runtime_error("Invalid JSON format: invalid array type.");
    }

    vec.clear();

    auto & array = std::get<JsonArray>(node->value);
    vec.resize(array.size());

    for (size_t i = 0; i < vec.size(); ++i)
    {
        // NOLINTNEXTLINE(misc-no-recursion)
        from_json_string(array[i], vec[i]); /* recursive here */
    }
}

} /* end namespace detail */

/// The macro to declare a class as serializable.
/// Use `register_member("key", class.member);` to add members when using this macro
/// The order of members in the JSON string is based on the order of `register_member` calls.
/// The access modifier of the members can be public or private.
/// The access modifier will be changed to private after the macro.
#define MM_DECL_SERIALIZABLE(...)                                                                   \
public:                                                                                             \
    std::string to_json() const override                                                            \
    {                                                                                               \
        std::ostringstream oss;                                                                     \
        oss << "{";                                                                                 \
        const char * separator = "";                                                                \
        auto register_member = [&](const char * name, auto && value)                                \
        {                                                                                           \
            oss << separator << "\"" << name << "\":" << detail::JsonHelper::to_json_string(value); \
            separator = ",";                                                                        \
        };                                                                                          \
        __VA_ARGS__                                                                                 \
        oss << "}";                                                                                 \
        return oss.str();                                                                           \
    }                                                                                               \
                                                                                                    \
    void from_json(const std::string & json) override                                               \
    {                                                                                               \
        auto jsonNode = std::make_unique<detail::JsonNode>(detail::JsonType::Object, json);         \
        auto register_member = [&](const char * name, auto && value)                                \
        {                                                                                           \
            if (jsonNode->type == detail::JsonType::Object)                                         \
            {                                                                                       \
                auto & obj = std::get<detail::JsonMap>(jsonNode->value);                            \
                auto it = obj.find(name);                                                           \
                if (it != obj.end())                                                                \
                {                                                                                   \
                    detail::JsonHelper::from_json_string(it->second, value);                        \
                }                                                                                   \
            }                                                                                       \
        };                                                                                          \
        __VA_ARGS__                                                                                 \
    }                                                                                               \
                                                                                                    \
private:                                                                                            \
    friend class detail::JsonHelper;                                                                \
                                                                                                    \
    /* for the nested object, we already parsed the json */                                         \
    void from_json(const detail::JsonMap & json_map)                                                \
    {                                                                                               \
        auto register_member = [&](const char * name, auto && value)                                \
        {                                                                                           \
            auto it = json_map.find(name);                                                          \
            if (it != json_map.end())                                                               \
            {                                                                                       \
                detail::JsonHelper::from_json_string(it->second, value);                            \
            }                                                                                       \
        };                                                                                          \
        __VA_ARGS__                                                                                 \
    } /* end MM_DECL_SERIALIZABLE*/

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
