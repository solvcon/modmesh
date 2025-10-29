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

#include <Qt>
#include <QDockWidget>
#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QPen>
#include <QPushButton>
#include <vector>

namespace modmesh
{

struct BoundingBox {
    QRect bbox;
    std::string label;
    float score;
};

class RVisionDockWidget;
class Impl; 

class RVisionImage : public QWidget
{
    Q_OBJECT
public:
    explicit RVisionImage(QWidget *parent = nullptr);
    void setDetections(const std::vector<BoundingBox>& boxes);
    void clearDetections();
    QImage currentImage() const;
    void loadImage(const QString& path);
protected:
    void paintEvent(QPaintEvent* event) override;
private:
    QImage m_image;
    std::vector<BoundingBox> m_boxes;
};

class RVisionDockWidget
    : public QDockWidget
{
    Q_OBJECT

public:
    explicit RVisionDockWidget(
        QString const & title = "Computer Vision",
        QWidget * parent = nullptr,
        Qt::WindowFlags flags = Qt::WindowFlags());

    QString command() const;

public slots:
    void setCommand(QString const & value);
    void executeCommand();
    void navigateCommand(int offset);

private:
    class Impl {
    public:
        QPushButton *toggleBtn = nullptr; 
        bool enableDetection = false;
    };
    std::unique_ptr<Impl> m_impl;
    RVisionImage * m_image = nullptr;
    void runYoloDetection(); 
}; /* end class RVisionDockWidget */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
