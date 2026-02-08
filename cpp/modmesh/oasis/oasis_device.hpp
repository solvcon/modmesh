#pragma once

/*
 * Copyright (c) 2026, Han-Xuan Huang <c1ydehhx@gmail.com>
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

#include <modmesh/base.hpp>
#include <pybind11/stl.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace modmesh
{

namespace oasis
{

static void append_signed_integer(std::vector<uint8_t> & segment, int value);
static void append_unsigned_integer(std::vector<uint8_t> & segment, int value);
static void append_magic_bytes(std::vector<uint8_t> & segment);

template <typename T>
void append_record_bytes(std::vector<uint8_t> bytes, T record);

/**
 * \class PolyRecord
 * \brief Convert Rect information to OASIS polygon record bytes.
 */
class PolyRecord
{
private:
    std::vector<std::pair<int, int>> vertexes;

public:
    explicit PolyRecord(std::vector<std::pair<int, int>> vertexes)
        : vertexes(std::move(vertexes)) {};
    PolyRecord(PolyRecord const &) = default;
    PolyRecord(PolyRecord &&) = default;
    PolyRecord & operator=(PolyRecord const &) = default;
    PolyRecord & operator=(PolyRecord &&) = default;
    ~PolyRecord() = default;

    std::vector<uint8_t> to_bytes() const;
};

/**
 * \class RectRecord
 * \brief Convert Rect information to OASIS rectangle record bytes.
 */
class RectRecord
{
private:
    std::pair<int, int> bottom_left;
    int w;
    int h;

public:
    RectRecord(std::pair<int, int> bottom_left, int w, int h)
        : bottom_left(bottom_left)
        , w(w)
        , h(h)
    {
    }
    RectRecord(RectRecord const &) = default;
    RectRecord(RectRecord &&) = default;
    RectRecord & operator=(RectRecord const &) = default;
    RectRecord & operator=(RectRecord &&) = default;
    ~RectRecord() = default;

    std::vector<uint8_t> to_bytes() const;
};

/**
 * \class OasisDevice
 * \brief OASIS device converts coordinates information to OASIS format.
 *
 * OasisDevice class store rectangles or polygons as OASIS format,
 * the implementation based on OASIS specification, convert coordinates
 * information (xy) to OASIS geometry format. The format are represented as
 * byte-continuations. LSB present that the next bytes is belonging the
 * group or not.
 */
class OasisDevice
{
private:
    std::vector<PolyRecord> polygon_records;
    std::vector<RectRecord> rect_records;

public:
    OasisDevice() = default;
    OasisDevice(OasisDevice const &) = default;
    OasisDevice(OasisDevice &&) = default;
    OasisDevice & operator=(OasisDevice const &) = default;
    OasisDevice & operator=(OasisDevice &&) = default;
    ~OasisDevice() = default;

    void add_poly_record(const PolyRecord & record);
    void add_rect_record(const RectRecord & record);

    std::vector<uint8_t> write();
};

} // namespace oasis

} // namespace modmesh

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: