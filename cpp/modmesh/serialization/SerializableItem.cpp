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

#include <modmesh/serialization/SerializableItem.hpp>

namespace modmesh
{

namespace detail
{

inline bool is_json_number(char c)
{
    return std::isdigit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
}

inline void throw_serialization_error(std::string const & message, int line, int column)
{
    throw std::runtime_error(std::format("{} (line: {}, column: {})", message, line, column));
}

std::string escape_string(std::string_view str_view)
{
    std::ostringstream oss;
    for (const char c : str_view)
    {
        switch (c)
        {
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            if (c < 32 || c >= 127)
            {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
            }
            else
            {
                oss << c;
            }
        }
    }
    return oss.str();
}

std::string trim_string(const std::string & str)
{
    const char * whitespace = " \t\n\r\f\v";
    const size_t start = str.find_first_not_of(whitespace);
    const size_t end = str.find_last_not_of(whitespace);
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

#define MM_DECL_CACULATE_LINE_COLUMN(CH) \
    if (CH == '\n')                      \
    {                                    \
        line += 1;                       \
        column = 1;                      \
    }                                    \
    else                                 \
    {                                    \
        column += 1;                     \
    }

// NOLINTBEGIN(readability-function-cognitive-complexity)
// NOLINTNEXTLINE(misc-no-recursion)
JsonArray JsonNode::parse_array(const std::string & json)
{
    JsonArray json_array;
    JsonState state = JsonState::Start;
    std::string value_expression;

    int depth = 0;

    int line = 1;
    int column = 1;

    if (json.empty())
    {
        throw_serialization_error("Invalid JSON format: empty JSON string.", 0, 0);
    }

    for (size_t index = 0; index < json.size(); index++)
    {
        const char c = json[index];

        if (is_whitespace(c))
        {
            MM_DECL_CACULATE_LINE_COLUMN(c)
            continue;
        }

        switch (state)
        {
        case JsonState::Start:
            if (index > 0 || c != '[')
            {
                throw_serialization_error("Invalid JSON format: missing opening bracket.", line, column);
            }

            state = JsonState::ObjectValue;
            break; /* end case JsonState::Start */
        case JsonState::ObjectValue:
            value_expression.clear();

            if (c == '{')
            {
                // go to the end of the object
                while (index < json.size())
                {
                    value_expression.push_back(json[index]);

                    if (json[index] == '{')
                    {
                        depth += 1;
                    }
                    else if (json[index] == '}')
                    {
                        depth -= 1;
                    }

                    if (depth == 0)
                    {
                        break;
                    }

                    MM_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }

                if (depth != 0)
                {
                    throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
                }

                // NOLINTNEXTLINE(misc-no-recursion)
                json_array.emplace_back(std::make_unique<JsonNode>(JsonType::Object, value_expression));
            }
            else if (c == '[')
            {
                // array in array is valid in JSON, but we do not support it yet
                throw_serialization_error("Invalid JSON format: unexpected array in array.", line, column);
            }
            else
            {
                if (!std::isalpha(c) && !is_json_number(c) && c != '"') // check if the value is a string, number, boolean, or null
                {
                    throw_serialization_error(std::format("Invalid JSON format: invalid value expression: {}",
                                                          value_expression),
                                              line,
                                              column);
                }

                // we assume the value is a string, number, boolean, or null, and the expression is correct
                // if the expression is not correct, the exception will be thrown when parsing the value later
                while (index < json.size() - 1)
                {
                    value_expression.push_back(json[index]);

                    const char c_next = json[index + 1];
                    if (c_next == ',' || c_next == ']')
                    {
                        break;
                    }

                    MM_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }

                value_expression = trim_string(value_expression);

                JsonType type = JsonType::Unknown;
                if (value_expression == "true" || value_expression == "false")
                {
                    type = JsonType::Boolean;
                }
                else if (value_expression == "null")
                {
                    type = JsonType::Null;
                }
                else if (value_expression[0] == '"' && value_expression[value_expression.size() - 1] == '"')
                {
                    type = JsonType::String;
                }
                else
                {
                    bool is_number = true;
                    for (const char c : value_expression)
                    {
                        if (!is_json_number(c))
                        {
                            is_number = false;
                            break;
                        }
                    }
                    if (is_number)
                    {
                        type = JsonType::Number;
                    }
                    else
                    {
                        throw_serialization_error(std::format("Invalid JSON format: invalid value expression: {}",
                                                              value_expression),
                                                  line,
                                                  column);
                    }
                }

                json_array.emplace_back(std::make_unique<JsonNode>(type, value_expression));
            }

            state = JsonState::Comma;

            break; /* end case JsonState::ObjectValue */
        case JsonState::Comma:
            if (c == ',')
            {
                state = JsonState::ObjectValue;
            }
            else if (c == ']')
            {
                state = JsonState::End;
            }
            break; /* end case JsonState::Comma */
        case JsonState::End:
            if (index != json.size() - 1)
            {
                throw_serialization_error("Invalid JSON format: extra characters after closing bracket.", line, column);
            }
            break; /* end case JsonState::End */
        case JsonState::ObjectKey:
            throw_serialization_error("Invalid JSON format: unexpected key in array.", line, column);
            break; /* end case JsonState::ObjectKey */
        case JsonState::Column:
            throw_serialization_error("Invalid JSON format: unexpected column in array.", line, column);
            break; /* end case JsonState::Column */
        }
    }

    if (state != JsonState::End)
    {
        throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
    }

    return json_array;
}
// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTBEGIN(readability-function-cognitive-complexity)
// NOLINTNEXTLINE(misc-no-recursion)
JsonMap JsonNode::parse_object(const std::string & json)
{
    JsonMap json_map;
    JsonState state = JsonState::Start;
    std::string key;
    std::string value_expression;

    int depth = 0;
    int line = 1;
    int column = 1;

    if (json.empty())
    {
        throw_serialization_error("Invalid JSON format: empty JSON string.", 0, 0);
    }

    for (size_t index = 0; index < json.size(); index++)
    {
        const char c = json[index];

        if (is_whitespace(c))
        {
            MM_DECL_CACULATE_LINE_COLUMN(c)
            continue;
        }

        switch (state)
        {
        case JsonState::Start:
            if (index > 0 || c != '{')
            {
                throw_serialization_error("Invalid JSON format: missing opening bracket.", line, column);
            }

            state = JsonState::ObjectKey;
            break; /* end case JsonState::Start */
        case JsonState::ObjectKey:
            if (c == '"')
            {
                key.clear();
                bool close = false; // get the key string directly

                index += 1;
                while (index < json.size())
                {
                    if (json[index] == '"')
                    {
                        close = true;
                        break;
                    }
                    key.push_back(json[index]);
                    MM_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }
                if (!close)
                {
                    throw_serialization_error("Invalid JSON format: missing closing quote for key.", line, column);
                }

                key = trim_string(key);
                state = JsonState::Column;
            }
            else
            {
                throw_serialization_error("Invalid JSON format: missing opening quote for key.", line, column);
            }
            break; /* end case JsonState::ObjectKey */
        case JsonState::Column:
            if (c == ':')
            {
                state = JsonState::ObjectValue;
            }
            break; /* end case JsonState::Column */
        case JsonState::ObjectValue:
            value_expression.clear();

            if (c == '{')
            {
                // go to the end of the object
                while (index < json.size())
                {
                    value_expression.push_back(json[index]);

                    if (json[index] == '{')
                    {
                        depth += 1;
                    }
                    else if (json[index] == '}')
                    {
                        depth -= 1;
                    }

                    if (depth == 0)
                    {
                        break;
                    }

                    MM_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }

                if (depth != 0)
                {
                    throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
                }

                // NOLINTNEXTLINE(misc-no-recursion)
                json_map.emplace(key, std::make_unique<JsonNode>(JsonType::Object, value_expression));
            }
            else if (c == '[')
            {
                // go to the end of the array
                while (index < json.size())
                {
                    value_expression.push_back(json[index]);

                    if (json[index] == '[')
                    {
                        depth += 1;
                    }
                    else if (json[index] == ']')
                    {
                        depth -= 1;
                    }

                    if (depth == 0)
                    {
                        break;
                    }

                    MM_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }

                if (depth != 0)
                {
                    throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
                }

                // NOLINTNEXTLINE(misc-no-recursion)
                json_map.emplace(key, std::make_unique<JsonNode>(JsonType::Array, value_expression));
            }
            else
            {
                if (!std::isalpha(c) && !is_json_number(c) && c != '"') // check if the value is a string, number, boolean, or null
                {
                    throw_serialization_error(std::format("Invalid JSON format: invalid value expression: {}",
                                                          value_expression),
                                              line,
                                              column);
                }

                // we assume the value is a string, number, boolean, or null, and the expression is correct
                // if the expression is not correct, the exception will be thrown when parsing the value later
                while (index < json.size() - 1)
                {
                    value_expression.push_back(json[index]);

                    const char c_next = json[index + 1];
                    if (c_next == ',' || c_next == '}')
                    {
                        break;
                    }

                    MM_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }

                value_expression = trim_string(value_expression);

                JsonType type = JsonType::Unknown;
                if (value_expression == "true" || value_expression == "false")
                {
                    type = JsonType::Boolean;
                }
                else if (value_expression == "null")
                {
                    type = JsonType::Null;
                }
                else if (value_expression[0] == '"' && value_expression[value_expression.size() - 1] == '"')
                {
                    type = JsonType::String;
                }
                else
                {
                    bool is_number = true;
                    for (const char c : value_expression)
                    {
                        if (!is_json_number(c))
                        {
                            is_number = false;
                            break;
                        }
                    }
                    if (is_number)
                    {
                        type = JsonType::Number;
                    }
                    else
                    {
                        throw_serialization_error(
                            std::format("Invalid JSON format: invalid value expression: {}", value_expression),
                            line,
                            column);
                    }
                }

                json_map.emplace(key, std::make_unique<JsonNode>(type, value_expression));
            }

            state = JsonState::Comma;

            break; /* end case JsonState::ObjectValue */
        case JsonState::Comma:
            if (c == ',')
            {
                state = JsonState::ObjectKey;
            }
            else if (c == '}')
            {
                state = JsonState::End;
            }
            break; /* end case JsonState::Comma */
        case JsonState::End:
            if (index != json.size() - 1)
            {
                throw_serialization_error("Invalid JSON format: extra characters after closing bracket.", line, column);
            }
            break; /* end case JsonState::End */
        }
    }

    if (state != JsonState::End)
    {
        throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
    }

    return json_map;
}
// NOLINTEND(readability-function-cognitive-complexity)

#undef MM_DECL_CACULATE_LINE_COLUMN

} /* end namespace detail */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
