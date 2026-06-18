#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/pilot/common_detail.hpp> // Must be the first include.

#include <QAction>

namespace modmesh
{

class RAction
    : public QAction
{
public:

    RAction(
        QString const & text,
        QString const & tipText,
        std::function<void(void)> callback,
        QObject * parent = nullptr);
}; /* end class RAction */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
