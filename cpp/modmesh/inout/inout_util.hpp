#pragma once

#include <regex>
#include <string>
#include <sstream>
#include <vector>

#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{

namespace inout
{

std::vector<std::string> tokenize(const std::string & str, const std::regex & regex_delim);
std::vector<std::string> tokenize(const std::string & str, const char delim);

} // namespace inout

} // namespace modmesh

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
