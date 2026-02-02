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

#include <modmesh/pilot/RPythonConsoleDockWidget.hpp>
#include <QVBoxLayout>
#include <QKeyEvent>
#include <QScrollBar>
#include <QTimer>

namespace modmesh
{

void RPythonHistoryTextEdit::mouseDoubleClickEvent(QMouseEvent * mouse_event)
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
    const QTextCursor cursor = textCursor();
    const bool is_first_line_focused = cursor.blockNumber() == 0;
    const bool is_last_line_focused = cursor.blockNumber() == document()->blockCount() - 1;

    if (Qt::Key_Return == event->key() || Qt::Key_Enter == event->key())
    {
        if (event->modifiers() & Qt::ShiftModifier)
        {
            this->insertPlainText("\n");
        }
        else
        {
            execute();
        }
    }
    else if (Qt::Key_Up == event->key() && is_first_line_focused)
    {
        navigate(/* offset */ -1);
    }
    else if (Qt::Key_Down == event->key() && is_last_line_focused)
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
    , m_scroll_area(new QScrollArea)
    , m_container(new QWidget)
    , m_history_edit(new RPythonHistoryTextEdit)
    , m_command_edit(new RPythonCommandTextEdit)
    , m_python_redirect(Toggle::instance().fixed().get_python_redirect())
{
    m_scroll_area->setWidgetResizable(true);
    setWidget(m_scroll_area);

    m_container->setLayout(new QVBoxLayout);
    m_scroll_area->setWidget(m_container);

    QPalette palette = QPalette();
    palette.setColor(QPalette::Base, Qt::white);
    palette.setColor(QPalette::Text, Qt::black);
    palette.setColor(QPalette::PlaceholderText, Qt::darkGray);

    m_history_edit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_history_edit->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_history_edit->setFont(QFont("Courier New"));
    m_history_edit->setReadOnly(true);
    m_history_edit->setPalette(palette);
    m_container->layout()->addWidget(m_history_edit);

    constexpr int commandEditMinHeight = 40;
    m_command_edit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_command_edit->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_command_edit->setFont(QFont("Courier New"));
    m_command_edit->setFixedHeight(commandEditMinHeight);
    m_command_edit->setPalette(palette);
    m_command_edit->setPlaceholderText("Shift+Enter to create new line. Enter to execute.");
    m_container->layout()->addWidget(m_command_edit);

    connect(
        m_history_edit->document(),
        &QTextDocument::contentsChanged,
        this,
        [this]()
        {
            const int newHeight = calcHeightToFitContents(m_history_edit);
            m_history_edit->setMinimumHeight(newHeight);
        });
    connect(
        m_command_edit->document(),
        &QTextDocument::contentsChanged,
        this,
        [this, commandEditMinHeight]()
        {
            const int newHeight = std::max(calcHeightToFitContents(m_command_edit), commandEditMinHeight);
            m_command_edit->setFixedHeight(newHeight);
        });
    connect(
        m_command_edit,
        &RPythonCommandTextEdit::cursorPositionChanged,
        this,
        [this]()
        {
            const QRect rect = m_command_edit->cursorRect();
            const QPoint point = m_command_edit->mapTo(m_container, rect.bottomRight());

            m_scroll_area->ensureVisible(point.x(), point.y(), 20, 20);
        });
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
    std::string const command = m_command_edit->toPlainText().trimmed().toStdString();

    if (command.empty())
    {
        return;
    }

    commitCommand(command);
    ++m_last_command_serial;

    const auto formatted_command = std::format("[{}] {}\n\n", m_last_command_serial, command);
    writeToHistory(formatted_command);

    m_command_edit->clear();
    m_current_command_index = static_cast<int>(m_committed_commands.size());

    auto & interp = modmesh::python::Interpreter::instance();
    m_python_redirect.activate();
    interp.exec_code(command);
    if (m_python_redirect.is_activated())
    {
        printCommandStdout(m_python_redirect.stdout_string());
        printCommandStderr(m_python_redirect.stderr_string());
    }
    m_python_redirect.deactivate();
}

void RPythonConsoleDockWidget::navigateCommand(int offset)
{
    int const commands_num = static_cast<int>(m_committed_commands.size()); // make msc happy.
    if (commands_num == m_current_command_index)
    {
        m_draft_command = m_command_edit->toPlainText().toStdString();
    }

    const int new_index = std::clamp(m_current_command_index + offset, 0, commands_num);

    const std::string & command_to_show = new_index == commands_num
                                              ? m_draft_command
                                              : m_committed_commands[new_index];

    m_current_command_index = new_index;
    setCommand(QString::fromStdString(command_to_show));
}

int RPythonConsoleDockWidget::calcHeightToFitContents(const QTextEdit * edit)
{
    edit->document()->setTextWidth(edit->viewport()->width());

    const int docH = static_cast<int>(std::ceil(edit->document()->size().height()));

    const int frame = 2 * edit->frameWidth();
    const int margins = edit->contentsMargins().top() + edit->contentsMargins().bottom();

    return docH + frame + margins;
}

void RPythonConsoleDockWidget::commitCommand(const std::string & command)
{
    m_committed_commands.push_back(command);

    if (m_committed_commands.size() > m_committed_commands_size_limit)
    {
        m_committed_commands.pop_front();
    }
}

void RPythonConsoleDockWidget::printCommandStdout(const std::string & stdout_message) const
{
    writeToHistory(stdout_message);
}

void RPythonConsoleDockWidget::printCommandStderr(const std::string & stderr_message) const
{
    writeToHistory(stderr_message);
}

void RPythonConsoleDockWidget::writeToHistory(const std::string & data) const
{
    QTextCursor cursor = m_history_edit->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
    m_history_edit->insertPlainText(QString::fromStdString(data));
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
