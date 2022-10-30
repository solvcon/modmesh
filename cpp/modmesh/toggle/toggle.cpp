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

#include <modmesh/toggle/toggle.hpp>
#ifdef MODMESH_METAL
#include <modmesh/device/metal/metal.hpp>
#endif // MODMESH_METAL

namespace modmesh
{

Toggle & Toggle::instance()
{
    static Toggle o;
    return o;
}

// NOLINTNEXTLINE(modernize-use-equals-default) lack of MODMESH_METAL
ProcessInfo::ProcessInfo()
{
#ifdef MODMESH_METAL
    modmesh::device::MetalManager::instance();
#endif // MODMESH_METAL
}

ProcessInfo & ProcessInfo::instance()
{
    static ProcessInfo inst;
    return inst;
}

void CommandLineInfo::populate(int argc, char ** argv, bool repopulate)
{
    if (!m_frozen && (!m_populated || repopulate))
    {
        m_populated_argv.clear();
        m_populated_argv.reserve(argc);
        size_t nchar = 0;
        for (int i = 0; i < argc; ++i)
        {
            m_populated_argv.emplace_back(argv[i]);
            nchar += m_populated_argv.back().size() + 1;
        }
        m_populated = true;
        // Determine Python entry-point mode.
        {
            // Populate char buffer.
            m_python_main_argv_char = SimpleArray<char>(nchar);
            m_python_main_argv_ptr = SimpleArray<char *>(argc);
            m_python_main_argc = 0;
            size_t ichar = 0;
            size_t icnt = 0;
            for (int i = 0; i < argc; ++i)
            {
                std::string const & arg = m_populated_argv[i];
                if ("--mode=python" == arg)
                {
                    m_python_main = true;
                }
                else
                {
                    ++m_python_main_argc;
                    m_python_main_argv_ptr[icnt++] = &m_python_main_argv_char[ichar];
                    for (size_t j = 0; j < arg.size(); ++j)
                    {
                        m_python_main_argv_char[ichar++] = arg[j];
                    }
                    m_python_main_argv_char[ichar++] = 0;
                }
            }
        }
        // Copy to other containers.
        m_python_argv = m_populated_argv;
        // Get executable base name.
        if (!m_populated_argv.empty())
        {
            std::string const & path = m_populated_argv.front();
            if (!path.empty())
            {
                size_t idx = path.find_last_of("/\\");
                idx = (idx < path.size()) ? idx + 1 : 0;
                m_executable_basename = path.substr(idx);
            }
        }
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
