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

#include <cstdlib>
#include <cmath>

namespace modmesh
{

int setenv(const char * name, const char * value, int overwrite)
{
#ifdef _WIN32
    int errcode = 0;
    if (!overwrite)
    {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize)
        {
            return errcode;
        }
    }
    return _putenv_s(name, value);
#else // _WIN32
    // NOLINTNEXTLINE(concurrency-mt-unsafe) TODO: make the wrapper thread safe.
    return ::setenv(name, value, overwrite);
#endif // _WIN32
}

Toggle & Toggle::instance()
{
    static Toggle o;
    return o;
}

SolidToggle::SolidToggle()
    : m_use_pyside(
#ifdef MODMESH_USE_PYSIDE
          true
#else
          false
#endif
      )
{
}

// NOLINTNEXTLINE(fuchsia-statically-constructed-objects,readability-redundant-string-init,cert-err58-cpp)
std::string const DynamicToggleTable::sentinel_string = "";

/* The macro gives debuggers a hard time. Manually expand it if you need to
 * step in a debugger. */
#define MM_DECL_DYNGET(CTYPE, MTYPE, SENTINEL)                                 \
    CTYPE DynamicToggleTable::get_##MTYPE(std::string const & key) const       \
    {                                                                          \
        auto it = m_key2index.find(key);                                       \
        if (it != m_key2index.end())                                           \
        {                                                                      \
            if (it->second.is_##MTYPE())                                       \
            {                                                                  \
                return m_vector_##MTYPE.at(it->second.index);                  \
            }                                                                  \
        }                                                                      \
        return SENTINEL;                                                       \
    }                                                                          \
    CTYPE HierarchicalToggleAccess::get_##MTYPE(std::string const & key) const \
    {                                                                          \
        return m_table->get_##MTYPE(rekey(key));                               \
    }
MM_DECL_DYNGET(bool, bool, false)
MM_DECL_DYNGET(int8_t, int8, 0)
MM_DECL_DYNGET(int16_t, int16, 0)
MM_DECL_DYNGET(int32_t, int32, 0)
MM_DECL_DYNGET(int64_t, int64, 0)
MM_DECL_DYNGET(double, real, std::nan("0"))
MM_DECL_DYNGET(std::string const &, string, sentinel_string)
#undef MM_DECL_DYNGET

/* The macro gives debuggers a hard time. Manually expand it if you need to
 * step in a debugger. */
#define MM_DECL_DYNSET(CTYPE, MTYPE, MTYPEC)                                                                                   \
    /* NOLINTNEXTLINE(bugprone-easily-swappable-parameters) */                                                                 \
    void DynamicToggleTable::set_##MTYPE(std::string const & key, CTYPE value)                                                 \
    {                                                                                                                          \
        auto it = m_key2index.find(key);                                                                                       \
        if (it != m_key2index.end())                                                                                           \
        {                                                                                                                      \
            if (it->second.is_##MTYPE())                                                                                       \
            {                                                                                                                  \
                m_vector_##MTYPE.at(it->second.index) = value;                                                                 \
            }                                                                                                                  \
            else                                                                                                               \
            {                                                                                                                  \
                /* do nothing */                                                                                               \
            }                                                                                                                  \
        }                                                                                                                      \
        else                                                                                                                   \
        {                                                                                                                      \
            DynamicToggleIndex const index{static_cast<uint32_t>(m_vector_##MTYPE.size()), DynamicToggleIndex::TYPE_##MTYPEC}; \
            m_key2index.insert({key, index});                                                                                  \
            m_vector_##MTYPE.push_back(value);                                                                                 \
        }                                                                                                                      \
    }                                                                                                                          \
    void HierarchicalToggleAccess::set_##MTYPE(std::string const & key, CTYPE value)                                           \
    {                                                                                                                          \
        m_table->set_##MTYPE(rekey(key), value);                                                                               \
    }
MM_DECL_DYNSET(bool, bool, BOOL)
MM_DECL_DYNSET(int8_t, int8, INT8)
MM_DECL_DYNSET(int16_t, int16, INT16)
MM_DECL_DYNSET(int32_t, int32, INT32)
MM_DECL_DYNSET(int64_t, int64, INT64)
MM_DECL_DYNSET(double, real, REAL)
MM_DECL_DYNSET(std::string const &, string, STRING)
#undef MM_DECL_DYNSET

HierarchicalToggleAccess HierarchicalToggleAccess::get_subkey(const std::string & key)
{
    return m_table->get_subkey(rekey(key));
}

void HierarchicalToggleAccess::add_subkey(const std::string & key)
{
    return m_table->add_subkey(rekey(key));
}

void DynamicToggleTable::add_subkey(std::string const & key)
{
    auto it = m_key2index.find(key);
    if (it == m_key2index.end())
    {
        DynamicToggleIndex const index{0, DynamicToggleIndex::TYPE_SUBKEY};
        m_key2index.insert({key, index});
    }
}

std::vector<std::string> DynamicToggleTable::keys() const
{
    std::vector<std::string> ret;
    ret.reserve(m_key2index.size());
    for (auto const & it : m_key2index)
    {
        ret.push_back(it.first);
    }
    return ret;
}

void DynamicToggleTable::clear()
{
    m_key2index.clear();
    m_vector_bool.clear();
    m_vector_int8.clear();
    m_vector_int16.clear();
    m_vector_int32.clear();
    m_vector_int64.clear();
    m_vector_real.clear();
    m_vector_string.clear();
}

// NOLINTNEXTLINE(modernize-use-equals-default) lack of MODMESH_METAL
ProcessInfo::ProcessInfo()
{
#ifdef MODMESH_METAL
    modmesh::device::MetalManager::instance();
#endif // MODMESH_METAL
}

ProcessInfo & ProcessInfo::set_environment_variables()
{
// TODO: Use Qt RHI when it stablizes.
// At the time of testing (Qt 6.4), RHI is not stable.  A workaround is to use
// OpenGL instead of RHI.  See more detail at
// https://doc.qt.io/qtforpython/overviews/qt3drender-porting-to-rhi.html
#if defined(QT3D_USE_RHI)
    setenv("QT3D_RENDERER", "rhi", 1);
#else
    setenv("QT3D_RENDERER", "opengl", 1);
#endif // QT3D_USE_RHI

    return *this;
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
