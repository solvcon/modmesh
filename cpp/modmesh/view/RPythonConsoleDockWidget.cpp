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

#include <modmesh/view/RPythonConsoleDockWidget.hpp>
#include <QVBoxLayout>
#include <QKeyEvent>

namespace modmesh
{

void RPythonConsoleDockWidget::appendPastCommand(const std::string & code)
{
    if (code.size() > 0)
    {
        m_past_command_strings.push_back(code);
        if (m_past_command_strings.size() > m_past_limit)
        {
            m_past_command_strings.pop_front();
        }
    }
}

void RPythonHistoryTextEdit::doubleClickHistoryEdit()
{
    // TODO: currently, it will select the whole content, could change the behavior
    QTextCursor cursor = textCursor();
    std::string text = toPlainText().toStdString();
    int startPos = 0;
    int endPos = static_cast<int>(text.size());
    cursor.setPosition(startPos);
    cursor.setPosition(endPos, QTextCursor::KeepAnchor);
    setTextCursor(cursor);
}

void RPythonCommandTextEdit::keyPressEvent(QKeyEvent * event)
{
    if (Qt::Key_Return == event->key() || Qt::Key_Enter == event->key())
    {
        execute();
    }
    else if (Qt::Key_Up == event->key())
    {
        navigate(/* offset */ -1);
    }
    else if (Qt::Key_Down == event->key())
    {
        navigate(/* offset */ 1);
    }
    else
    {
        QTextEdit::keyPressEvent(event);
    }
}

RPythonConsoleDockWidget::RPythonConsoleDockWidget(const QString & title, QWidget * parent, Qt::WindowFlags flags)
    : QDockWidget(title, parent, flags)
    , m_history_edit(new RPythonHistoryTextEdit)
    , m_command_edit(new RPythonCommandTextEdit)
{
    setWidget(new QWidget);
    widget()->setLayout(new QVBoxLayout);

    m_history_edit->setFont(QFont("Courier New"));
    m_history_edit->setHtml(QString(""));
    m_history_edit->setReadOnly(true);
    widget()->layout()->addWidget(m_history_edit);

    m_command_edit->setFont(QFont("Courier New"));
    m_command_edit->setPlainText(QString(""));
    m_command_edit->setFixedHeight(40);
    widget()->layout()->addWidget(m_command_edit);

    connect(m_command_edit, &RPythonCommandTextEdit::execute, this, &RPythonConsoleDockWidget::executeCommand);
    connect(m_command_edit, &RPythonCommandTextEdit::navigate, this, &RPythonConsoleDockWidget::navigateCommand);
}

QString RPythonConsoleDockWidget::command() const
{
    return m_command_edit->toPlainText();
}

void RPythonConsoleDockWidget::setCommand(const QString & value)
{
    m_command_edit->setPlainText(value);
    m_command_string = value.toStdString();
    // Move cursor to the end of line.
    {
        QTextCursor cursor = m_command_edit->textCursor();
        cursor.movePosition(QTextCursor::EndOfLine);
        m_command_edit->setTextCursor(cursor);
    }
    if (!m_command_edit->hasFocus())
    {
        m_command_edit->setFocus();
    }
}

void RPythonConsoleDockWidget::executeCommand()
{
    std::string const code = m_command_edit->toPlainText().toStdString();
    appendPastCommand(code);
    printCommandHistory();
    m_command_edit->setPlainText("");
    m_command_string = "";
    m_current_command_index = static_cast<int>(m_past_command_strings.size());
    auto & interp = modmesh::python::Interpreter::instance();

    interp.exec_code(m_config["appEnvName"], code);
    printCommandOutput();
}

void RPythonConsoleDockWidget::printCommandOutput()
{
    const std::string redirect_stdout_file_path = m_config["redirectStdOutFile"];
    const std::string redirect_stderr_file_path = m_config["redirectStdErrFile"];

    std::string stdout_str;
    std::string stderr_str;
    std::ifstream redirect_stdout_stream(redirect_stdout_file_path);
    std::ifstream redirect_stderr_stream(redirect_stderr_file_path);
    if (!redirect_stdout_stream.bad())
    {
        Formatter formatter;
        std::string line;
        while (std::getline(redirect_stdout_stream, line))
        {
            formatter << line << "<br>";
        }
        stdout_str = formatter.str();
        redirect_stdout_stream.close();
    }
    else
    {
        std::string err_msg = "cannot open file: " + redirect_stdout_file_path;
        throw std::runtime_error(err_msg);
    }
    if (!redirect_stderr_stream.bad())
    {
        Formatter formatter;
        std::string line;
        while (std::getline(redirect_stderr_stream, line))
        {
            formatter << line << "<br>";
        }
        stderr_str = formatter.str();
        redirect_stdout_stream.close();
    }
    else
    {
        std::string err_msg = "cannot open file: " + redirect_stderr_file_path;
        throw std::runtime_error(err_msg);
    }

    printCommandStdout(stdout_str);
    printCommandStderr(stderr_str);
}

void RPythonConsoleDockWidget::printCommandHistory()
{
    QTextCursor cursor = m_history_edit->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
    m_history_edit->insertHtml(m_commandHtml + m_command_edit->toPlainText() + m_endHtml);
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
}

void RPythonConsoleDockWidget::printCommandStdout(const std::string & stdout_message)
{
    QTextCursor cursor = m_history_edit->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
    m_history_edit->insertHtml(m_stdoutHtml + QString::fromStdString(stdout_message) + m_endHtml);
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
}

void RPythonConsoleDockWidget::printCommandStderr(const std::string & stderr_message)
{
    QTextCursor cursor = m_history_edit->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
    m_history_edit->insertHtml(m_stderrHtml + QString::fromStdString(stderr_message) + m_endHtml);
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
}

void RPythonConsoleDockWidget::navigateCommand(int offset)
{
    int const cmdsize = static_cast<int>(m_past_command_strings.size()); // make msc happy.
    if (cmdsize == m_current_command_index)
    {
        m_command_string = m_command_edit->toPlainText().toStdString();
    }

    int new_index = m_current_command_index + offset;
    if (new_index > cmdsize)
    {
        new_index = cmdsize;
    }
    else if (new_index < 0)
    {
        new_index = 0;
    }

    if ((0 == m_past_command_strings.size()) || (new_index == m_current_command_index))
    {
        // do nothing
    }
    else
    {
        if (new_index > m_current_command_index)
        {
            if (new_index >= static_cast<int>(m_past_command_strings.size()))
            {
                m_command_edit->setPlainText(QString::fromStdString(m_command_string));
            }
            else
            {
                m_command_edit->setPlainText(QString::fromStdString(m_past_command_strings[new_index]));
            }
            m_current_command_index = new_index;
        }
        else // new_index < m_current_command_index
        {
            m_command_edit->setPlainText(QString::fromStdString(m_past_command_strings[new_index]));
            m_current_command_index = new_index;
        }
        // Move cursor to the end of line.
        {
            QTextCursor cursor = m_command_edit->textCursor();
            cursor.movePosition(QTextCursor::EndOfLine);
            m_command_edit->setTextCursor(cursor);
        }
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
