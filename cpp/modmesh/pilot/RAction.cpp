/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/pilot/RAction.hpp> // Must be the first include.

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

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
