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

#include <cstdint>
#include <utility>
#include <vector>

namespace modmesh
{

class OasisRecordPoly;
class OasisRecordRect;

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
public:
    static void append_signed_integer(std::vector<uint8_t> & segment, int value);
    static void append_unsigned_integer(std::vector<uint8_t> & segment, int value);

    OasisDevice() = default;
    OasisDevice(OasisDevice const &) = default;
    OasisDevice(OasisDevice &&) = default;
    OasisDevice & operator=(OasisDevice const &) = default;
    OasisDevice & operator=(OasisDevice &&) = default;
    ~OasisDevice() = default;

    void add_poly_record(const OasisRecordPoly & record);
    void add_rect_record(const OasisRecordRect & record);

    std::vector<uint8_t> to_bytes();

private:
    static void append_magic_bytes(std::vector<uint8_t> & segment);
    static void append_start_record_bytes(std::vector<uint8_t> & segment);
    static void append_cell_and_cell_name_record_byte(std::vector<uint8_t> & segment);
    static void append_end_record_byte(std::vector<uint8_t> & segment);

    template <typename T>
    static void append_record_bytes(std::vector<uint8_t> & bytes, const T & record);

    std::vector<OasisRecordPoly> m_polygon_records;
    std::vector<OasisRecordRect> m_rect_records;
}; /* end class OasisDevice */

/**
 * \class PolyRecord
 * \brief Convert Rect information to OASIS polygon record bytes.
 */
class OasisRecordPoly
{

public:
    explicit OasisRecordPoly(std::vector<std::pair<int, int>> vertices)
        : m_vertices(std::move(vertices)) {};
    OasisRecordPoly() = delete;
    OasisRecordPoly(OasisRecordPoly const &) = default;
    OasisRecordPoly(OasisRecordPoly &&) = default;
    OasisRecordPoly & operator=(OasisRecordPoly const &) = default;
    OasisRecordPoly & operator=(OasisRecordPoly &&) = default;
    ~OasisRecordPoly() = default;

    std::vector<uint8_t> to_bytes() const;

private:
    std::vector<std::pair<int, int>> m_vertices;
}; /* end class OasisRecordPoly */

/**
 * \class RectRecord
 * \brief Convert Rect information to OASIS rectangle record bytes.
 */
class OasisRecordRect
{
public:
    OasisRecordRect(int left, int lower, int w, int h)
        : m_left(left)
        , m_lower(lower)
        , m_w(w)
        , m_h(h)
    {
    }
    OasisRecordRect() = delete;
    OasisRecordRect(OasisRecordRect const &) = default;
    OasisRecordRect(OasisRecordRect &&) = default;
    OasisRecordRect & operator=(OasisRecordRect const &) = default;
    OasisRecordRect & operator=(OasisRecordRect &&) = default;
    ~OasisRecordRect() = default;

    std::vector<uint8_t> to_bytes() const;

private:
    int m_left, m_lower, m_w, m_h;
}; /* end class OasisRecordRect */

} // namespace modmesh

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: