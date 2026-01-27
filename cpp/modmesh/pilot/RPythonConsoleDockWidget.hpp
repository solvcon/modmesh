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
#include <stdexcept>
#include <fstream>

#include <Qt>
#include <QDockWidget>
#include <QScrollArea>
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

class RPythonHistoryTextEdit
    : public QTextEdit
{
    Q_OBJECT

    void mouseDoubleClickEvent(QMouseEvent *) override;
}; /* end class RPythonHistoryTextEdit */

class RPythonConsoleDockWidget
    : public QDockWidget
{
    Q_OBJECT

public:

    explicit RPythonConsoleDockWidget(
        QString const & title = "Console",
        QWidget * parent = nullptr,
        Qt::WindowFlags flags = Qt::WindowFlags());

    QString command() const;
    void setCommand(QString const & value);

    bool hasPythonRedirect() const { return m_python_redirect.is_enabled(); }

    RPythonConsoleDockWidget & setPythonRedirect(bool enabled)
    {
        m_python_redirect.set_enabled(enabled);
        return *this;
    }

    void writeToHistory(std::string const & data) const;

public slots:
    void executeCommand();
    void navigateCommand(int offset);

private:
    static int calcHeightToFitContents(const QTextEdit * edit);

    void commitCommand(std::string const & command);
    void printCommandStdout(const std::string & stdout_message) const;
    void printCommandStderr(const std::string & stderr_message) const;

    QScrollArea * m_scroll_area = nullptr;
    QWidget * m_container = nullptr;
    RPythonHistoryTextEdit * m_history_edit = nullptr;
    RPythonCommandTextEdit * m_command_edit = nullptr;
    std::string m_draft_command;
    std::deque<std::string> m_committed_commands;
    size_t m_committed_commands_size_limit = 1024;
    int m_current_command_index = 0;
    int m_last_command_serial = 0;

    python::PythonStreamRedirect m_python_redirect;
}; /* end class RPythonConsoleDockWidget */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
