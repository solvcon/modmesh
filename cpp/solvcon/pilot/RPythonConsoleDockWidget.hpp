#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/python/common.hpp> // must be first.

#include <string>
#include <deque>
#include <stdexcept>
#include <fstream>

#include <Qt>
#include <QCompleter>
#include <QDockWidget>
#include <QStringListModel>
#include <QTextEdit>

namespace solvcon
{

class RPythonCommandTextEdit
    : public QTextEdit
{
    Q_OBJECT

public:

    void setCompleter(QCompleter * completer);
    QCompleter * completer() const { return m_completer; }
    QString completionPrefix() const;

    void keyPressEvent(QKeyEvent * event) override;

signals:

    void execute();
    void navigate(int offset);
    void completionRequested(const QString & prefix);

private slots:

    void insertCompletion(const QString & completion);

private:

    QCompleter * m_completer = nullptr;

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

private slots:
    void handleCompletionRequest(const QString & prefix);
    void updateCompletionPrefix();

private:
    static int calcHeightToFitContents(const QTextEdit * edit);

    void commitCommand(std::string const & command);
    void printCommandStdout(const std::string & stdout_message) const;
    void printCommandStderr(const std::string & stderr_message) const;

    RPythonHistoryTextEdit * m_history_edit = nullptr;
    RPythonCommandTextEdit * m_command_edit = nullptr;
    std::string m_draft_command;
    std::deque<std::string> m_committed_commands;
    size_t m_committed_commands_size_limit = 1024;
    int m_current_command_index = 0;
    int m_last_command_serial = 0;

    python::PythonStreamRedirect m_python_redirect;

    QCompleter * m_completer = nullptr;
    QStringListModel * m_completer_model = nullptr;
    QString m_completer_root_prefix;
}; /* end class RPythonConsoleDockWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
