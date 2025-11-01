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

#include <modmesh/pilot/RVisionDockWidget.hpp>
#include <QVBoxLayout>
#include <QKeyEvent>
#include <QPushButton>
#include <QFileDialog>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <QImage>

namespace modmesh
{

RVisionDockWidget::RVisionDockWidget(const QString & title, QWidget * parent, Qt::WindowFlags flags)
    : QDockWidget(title, parent, flags)
    , m_image(new RVisionImage)
    , m_impl(new Impl)
{
    setWidget(new QWidget);
    widget()->setLayout(new QVBoxLayout);

    m_image->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_image->setMinimumSize(200, 200);
    m_image->setMaximumSize(800, 800);
    widget()->layout()->addWidget(m_image);

    QPushButton *loadBtn = new QPushButton("Load Image");
    widget()->layout()->addWidget(loadBtn);
    connect(loadBtn, &QPushButton::clicked, this, [this]() {
        QString fileName = QFileDialog::getOpenFileName(this, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");
        if (!fileName.isEmpty()) {
            m_image->loadImage(fileName);
            if (m_impl->enableDetection) {
                runYoloDetection();
            }
        }
    });

    m_impl->toggleBtn = new QPushButton("Enable YOLO Detection");
    widget()->layout()->addWidget(m_impl->toggleBtn);
    connect(m_impl->toggleBtn, &QPushButton::clicked, this, [this]() {
        m_impl->enableDetection = !m_impl->enableDetection;
        if (m_impl->enableDetection) {
            m_impl->toggleBtn->setText("Disable YOLO Detection");
            runYoloDetection();
        } else {
            m_impl->toggleBtn->setText("Enable YOLO Detection");
            m_image->clearDetections();
        }
    });
}

void RVisionDockWidget::setCommand(QString const &)
{
    // TODO: Implement functionality
}

void RVisionDockWidget::executeCommand()
{
    // TODO: Implement functionality
}

void RVisionDockWidget::navigateCommand(int)
{
    // TODO: Implement functionality
}

void RVisionDockWidget::runYoloDetection()
{
    if (!m_impl->enableDetection) return;
    QImage img = m_image->currentImage();
    if (img.isNull()) return;

    namespace py = pybind11;
    py::gil_scoped_acquire gil;
    int width = img.width();
    int height = img.height();
    int channels = 3;
    QImage rgbImg = img.convertToFormat(QImage::Format_RGB888);
    const uchar *data = rgbImg.bits();
    py::array_t<uint8_t> np_img({height, width, channels}, data);
    py::object vision_mod = py::module_::import("modmesh.pilot._vision");
    py::object yolo_func = vision_mod.attr("yolo_detect");
    py::object result = yolo_func(np_img);
    std::vector<BoundingBox> boxes;
    py::list result_list = result;
    for (auto item : result_list) {
        py::dict d = py::reinterpret_borrow<py::dict>(item);
        std::vector<int> bbox;
        py::list bbox_list = d["bbox"];
        for (auto v : bbox_list) {
            bbox.push_back(v.cast<int>());
        }
        std::string label = d["label"].cast<std::string>();
        float score = d["score"].cast<float>();
        if (bbox.size() == 4) {
            QRect rect(bbox[0], bbox[1], bbox[2], bbox[3]);
            boxes.push_back({rect, label, score});
        }
    }
    m_image->setDetections(boxes);
}

RVisionImage::RVisionImage(QWidget *parent)
    : QWidget(parent)
{
    m_image.load(":/default.jpg");
}
void RVisionImage::setDetections(const std::vector<BoundingBox>& boxes) {
    m_boxes = boxes;
    update();
}
void RVisionImage::clearDetections() {
    m_boxes.clear();
    update();
}
QImage RVisionImage::currentImage() const { return m_image; }
void RVisionImage::loadImage(const QString& path) {
    m_image.load(path);
    update();
}
void RVisionImage::paintEvent(QPaintEvent* event) {
    QWidget::paintEvent(event);
    QPainter painter(this);
    painter.drawImage(rect(), m_image);
    painter.setPen(QPen(Qt::red, 2));
    for (const auto& box : m_boxes) {
        painter.drawRect(box.bbox);
        painter.drawText(box.bbox.topLeft(), QString::fromStdString(box.label));
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
