#pragma once

#include <regex>
#include <string>
#include <sstream>

#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{
namespace inout
{
small_vector<std::string> tokenize(const std::string & str, const std::string delim);
small_vector<std::string> tokenize(const std::string & str, const char delim);
} // namespace inout
} // namespace modmesh

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
