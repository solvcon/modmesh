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

#include <modmesh/view/RAction.hpp> // Must be the first include.

#include <functional>

namespace modmesh
{

RAction::RAction(QString const & text, QString const & tipText, std::function<void(void)> callback, QObject * parent)
    : QAction(text, parent)
{
    setStatusTip(tipText);
    if (callback != nullptr)
    {
        connect(this, &QAction::triggered, this, callback);
    }
}

RPythonAction::RPythonAction(const QString & text, const QString & tipText, const QString & pyFuncName, QObject * parent)
    : QAction(text, parent)
    , m_pyFuncName(pyFuncName)
{
    setStatusTip(tipText);
    connect(this, &QAction::triggered, this, &RPythonAction::run);
}

void RPythonAction::run()
{
    std::string const fullname = m_pyFuncName.toStdString();
    std::string::size_type const pos = fullname.rfind('.');
    std::string modname, funcname;
    if (pos == std::string::npos)
    {
        std::cerr << fullname << "does not have module" << std::endl;
    }
    else
    {
        modname = fullname.substr(0, pos);
        funcname = fullname.substr(pos + 1);
    }

    namespace py = pybind11;
    try
    {
        py::module_ appmod = py::module_::import(modname.c_str());
        appmod.attr(funcname.c_str())();
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
}

RAppAction::RAppAction(const QString & text, const QString & tipText, const QString & appName, QObject * parent)
    : QAction(text, parent)
    , m_appName(appName)
{
    setStatusTip(tipText);
    connect(this, &QAction::triggered, this, &RAppAction::run);
}

void RAppAction::run()
{
    namespace py = pybind11;
    try
    {
        py::module_ appmod = py::module_::import(m_appName.toStdString().c_str());
        appmod.reload();
        appmod.attr("load_app")();
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
