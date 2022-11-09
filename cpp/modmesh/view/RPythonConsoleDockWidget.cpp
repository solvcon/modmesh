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

void RPythonCommandTextEdit::keyPressEvent(QKeyEvent * event)
{
    if (Qt::Key_Return == event->key())
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
    , m_history_edit(new QTextEdit)
    , m_command_edit(new RPythonCommandTextEdit)
{
    setWidget(new QWidget);
    widget()->setLayout(new QVBoxLayout);

    m_history_edit->setFont(QFont("Courier New"));
    m_history_edit->setPlainText(QString(""));
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
    m_history_edit->insertPlainText(m_command_edit->toPlainText());
    m_history_edit->insertPlainText("\n");
    m_command_edit->setPlainText("");
    m_command_string = "";
    m_current_command_index = static_cast<int>(m_past_command_strings.size());
    auto & interp = modmesh::python::Interpreter::instance();
    interp.exec_code(code);
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
