#include <modmesh/inout/inout_util.hpp>

namespace modmesh
{
namespace inout
{
small_vector<std::string> tokenize(const std::string & str, const std::string delim)
{
    small_vector<std::string> output;
    std::regex regex_delim(delim);
    std::sregex_iterator iter(str.begin(), str.end(), regex_delim);
    std::sregex_iterator end;
    while (iter != end)
    {
        output.push_back(iter->str());
        ++iter;
    }
    return output;
}
small_vector<std::string> tokenize(const std::string & str, const char delim)
{
    small_vector<std::string> output;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim))
    {
        output.push_back(token);
    }
    return output;
}
} // namespace inout
} // namespace modmesh
