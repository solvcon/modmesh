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

#include <modmesh/python/common.hpp> // must be first.

#include <string>
#include <deque>

#include <Qt>
#include <QDockWidget>
#include <QTextEdit>

namespace modmesh
{

class RPythonCommandTextEdit
    : public QTextEdit
{
    Q_OBJECT

public:

    void keyPressEvent(QKeyEvent * event) override;

signals:

    void execute();
    void navigate(int offset);

}; /* end class RPythonCommandTextEdit */

class RPythonConsoleDockWidget
    : public QDockWidget
{
    Q_OBJECT

public:

    explicit RPythonConsoleDockWidget(
        QString const & title = "Console",
        QWidget * parent = nullptr,
        Qt::WindowFlags flags = Qt::WindowFlags(),
        std::shared_ptr<RPythonConsole> console = nullptr);

    QString command() const;

public slots:

    void setCommand(QString const & value);
    void executeCommand();
    void navigateCommand(int offset);

private:

    void appendPastCommand(std::string const & code);

    QTextEdit * m_history_edit = nullptr;
    RPythonCommandTextEdit * m_command_edit = nullptr;
    std::string m_command_string;
    std::deque<std::string> m_past_command_strings;
    int m_current_command_index = 0;
    size_t m_past_limit = 1024;

}; /* end class RPythonConsoleDockWidget */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
