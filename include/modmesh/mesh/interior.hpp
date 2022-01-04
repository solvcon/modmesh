#pragma once

/*
 * Copyright (c) 2021, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/mesh/StaticMesh_decl.hpp>

namespace modmesh
{

/**
 * Extract interier faces from node list of cells.  Subroutine is designed to
 * handle all types of cells.
 */
template < typename D /* derived type */, uint8_t ND >
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
void StaticMeshBase<D, ND>::build_faces_from_cells()
{

    size_t const mface = std::accumulate
    (
        m_cltpn.begin() + m_cltpn.nghost(), m_cltpn.end(), 0
      , [](size_t a, int8_t b){ return a + CellType::by_id(b).nface(); }
    );

    // create temporary tables.
    SimpleArray<int_type> tclfcs(small_vector<size_t>{m_ncell, CLMFC+1}, -1);
    SimpleArray<int_type> tfctpn(small_vector<size_t>{mface}, -1);
    SimpleArray<int_type> tfcnds(small_vector<size_t>{mface, FCMND+1}, -1);
    SimpleArray<int_type> tfccls(small_vector<size_t>{mface, FCREL}, -1);

    // extract face definition from the node list of cells.
    int_type * pclnds = m_clnds.body();
    int_type * pclfcs = tclfcs.body();
    int_type * pfctpn = tfctpn.body();
    int_type * pfcnds = tfcnds.body();
    int_type ifc = 0;
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        int_type const tpnicl = m_cltpn(icl);
        // parse each type of cell.
        if (tpnicl == 0 || tpnicl == 1)
        {
            // do nothing.
        }
        else if (tpnicl == 2) // line.
        {
            // extract 2 points from a line.
            pclfcs[0] = 2;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 0; // face type is point.
                pfcnds[it*(FCMND+1)] = 1;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 3) // quadrilateral.
        {
            // extract 4 lines from a quadrilateral.
            pclfcs[0] = 4;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 1; // face type is line.
                pfcnds[it*(FCMND+1)] = 2;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 4) // triangle.
        {
            // extract 3 lines from a triangle.
            pclfcs[0] = 3;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 1; // face type is line.
                pfcnds[it*(FCMND+1)] = 2;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 5) // hexahedron.
        {
            // extract 6 quadrilaterals from a hexahedron.
            pclfcs[0] = 6;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[7];
            pfcnds[4] = pclnds[6];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[5];
            pfcnds[2] = pclnds[6];
            pfcnds[3] = pclnds[7];
            pfcnds[4] = pclnds[8];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[8];
            pfcnds[4] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[5];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 6.
            pclfcs[6] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[8];
            pfcnds[4] = pclnds[7];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 6) // tetrahedron.
        {
            // extract 4 triangles from a tetrahedron.
            pclfcs[0] = 4;
            for (int_type it = 0 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 7) // prism.
        {
            // extract 2 triangles and 3 quadrilaterals from a prism.
            pclfcs[0] = 5;
            for (int_type it = 0 ; it < 2 ; ++it)
            {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            }
            for (int_type it = 2 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[6];
            pfcnds[3] = pclnds[5];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[5];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        else if (tpnicl == 8) // pyramid.
        {
            // extract 4 triangles and 1 quadrilaterals from a pyramid.
            pclfcs[0] = 5;
            for (int_type it = 0 ; it < 4 ; ++it)
            {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            }
            for (int_type it = 4 ; it < pclfcs[0] ; ++it)
            {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            }
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
        }
        // advance pointers.
        pclnds += CLMND+1;
        pclfcs += CLMFC+1;
    }

    // build the hash table, to know what faces connect to each node.
    /// first pass: get the maximum number of faces.
    std::vector<size_t> ndnfc(nnode(), 0);
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type const * p = tfcnds.vptr(ifc, 1) ; p != tfcnds.vptr(ifc, tfcnds(ifc, 0)+1) ; ++p)
        {
            ndnfc[*p] += 1;
        }
    }
    size_t const ndmfc = *std::max_element(ndnfc.begin(), ndnfc.end());
    /// second pass: scan again to build hash table.
    SimpleArray<int_type> ndfcs(small_vector<size_t>{nnode(), static_cast<size_t>(ndmfc)+1});
    for (size_t ind = 0 ; ind < nnode() ; ++ind)
    {
        ndfcs(ind, 0) = 0;
        for (size_t it = 1 ; it <= ndmfc ; ++it)
        {
            ndfcs(ind, it) = -1;
        }
    }
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type const * p = tfcnds.vptr(ifc, 1) ; p != tfcnds.vptr(ifc, tfcnds(ifc, 0)+1) ; ++p)
        {
            ndfcs(*p, 0) += 1;
            ndfcs(*p, ndfcs(*p, 0)) = ifc;
        }
    }

    // scan for duplicated faces and build duplication map.
    std::vector<size_t> map(mface);
    for (size_t ifc = 0 ; ifc < mface ; ++ifc) // initialize.
    {
        map[ifc] = ifc;
    }
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        if (map[ifc] == ifc)
        {
            int_type const nd1 = tfcnds(ifc, 1);    // take only the FIRST node of a face.
            for (int_type it = 1 ; it <= ndfcs(nd1, 0) ; ++it)
            {
                size_t const jfc = ndfcs(nd1, it);
                // test for duplication.
                if ((jfc != ifc) && (tfctpn(jfc) == tfctpn(ifc)))
                {
                    int_type cond = tfcnds(jfc, 0);
                    // scan all nodes in ifc and jfc to see if all the same.
                    for (int_type jnf = 1 ; jnf <= tfcnds(jfc, 0) ; ++jnf)
                    {
                        for (int_type inf = 1 ; inf <= tfcnds(ifc, 0) ; ++inf)
                        {
                            if (tfcnds(jfc, jnf) == tfcnds(ifc, inf))
                            {
                                cond -= 1;
                                break;
                            }
                        }
                    }
                    if (cond == 0)
                    {
                        map[jfc] = ifc;  // record duplication.
                    }
                }
            }
        }
    }

    // use the duplication map to remap nodes in faces, and build renewed map.
    std::vector<int_type> map2(mface);
    {
        size_t jfc = 0;
        for (size_t ifc = 0 ; ifc < mface ; ++ifc)
        {
            if (map[ifc] == ifc)
            {
                for (int_type inf = 0 ; inf <= FCMND ; ++inf)
                {
                    tfcnds(jfc, inf) = tfcnds(ifc, inf);
                }
                tfctpn(jfc) = tfctpn(ifc);
                map2[ifc] = jfc;
                // increment j-face.
                jfc += 1;
            }
            else
            {
                map2[ifc] = map2[map[ifc]];
            }
        }
        m_nface = jfc; // record deduplicated number of face.
    }

    // rebuild faces in cells and build face neighboring, according to the
    // renewed face map.
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        for (int_type ifl = 1 ; ifl <= tclfcs(icl, 0) ; ++ifl)
        {
            int_type const ifc = tclfcs(icl, ifl);
            int_type const jfc = map2[ifc];
            // rebuild faces in cells.
            tclfcs(icl, ifl) = jfc;
            // build face neighboring.
            if      (-1 == tfccls(jfc, 0)) { tfccls(jfc, 0) = icl; }
            else if (-1 == tfccls(jfc, 1)) { tfccls(jfc, 1) = icl; }
        }
    }

    // recreate member tables.
    m_fctpn.swap(SimpleArray<int_type>(small_vector<size_t>{m_nface}));
    std::copy(tfctpn.vptr(0), tfctpn.vptr(m_nface), m_fctpn.vptr(0));
    m_fcnds.swap(SimpleArray<int_type>(small_vector<size_t>{m_nface, FCMND+1}));
    std::copy(tfcnds.vptr(0, 0), tfcnds.vptr(m_nface, 0), m_fcnds.vptr(0, 0));
    m_fccls.swap(SimpleArray<int_type>(small_vector<size_t>{m_nface, FCREL}));
    std::copy(tfccls.vptr(0, 0), tfccls.vptr(m_nface, 0), m_fccls.vptr(0, 0));
    m_fccnd.swap(SimpleArray<real_type>(small_vector<size_t>{nface(), ND}, 0));
    m_fcnml.swap(SimpleArray<real_type>(small_vector<size_t>{nface(), ND}, 0));
    m_fcara.swap(SimpleArray<real_type>(small_vector<size_t>{nface()}, 0));
    std::copy(tclfcs.vptr(0, 0), tclfcs.vptr(m_ncell, 0), m_clfcs.vptr(0, 0));

}

/**
 * Calculate all metric information, including:
 *
 *  1. center of faces.
 *  2. unit normal and area of faces.
 *  3. center of cells.
 *  4. volume of cells.
 *
 * And fcnds could be reordered.
 */
template < typename D /* derived type */, uint8_t ND >
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
void StaticMeshBase<D, ND>::calc_metric()
{
    // compute face centroids.
    if (NDIM == 2)
    {
        // 2D faces must be edge.
        for (size_t ifc = 0 ; ifc < m_nface ; ++ifc)
        {
            // point 1.
            {
                int_type const ind = m_fcnds(ifc, 1);
                m_fccnd(ifc, 0) = m_ndcrd(ind, 0);
                m_fccnd(ifc, 1) = m_ndcrd(ind, 1);
            }
            // point 2.
            {
                int_type const ind = m_fcnds(ifc, 2);
                m_fccnd(ifc, 0) += m_ndcrd(ind, 0);
                m_fccnd(ifc, 1) += m_ndcrd(ind, 1);
            }
            // average.
            m_fccnd(ifc, 0) /= 2;
            m_fccnd(ifc, 1) /= 2;
        }
    }
    else if (NDIM == 3)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            std::array<real_type, NDIM> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
            std::array<std::array<real_type, NDIM>, FCMND+2> cfd; // NOLINT(cppcoreguidelines-pro-type-member-init)
            // find averaged point.
            cfd[0][0] = cfd[0][1] = cfd[0][2] = 0.0;
            size_t const nnd = m_fcnds(ifc, 0);
            for (size_t inf = 1 ; inf <= nnd ; ++inf)
            {
                int_type const ind = m_fcnds(ifc, inf);
                cfd[inf][0]  = m_ndcrd(ind, 0);
                cfd[0  ][0] += m_ndcrd(ind, 0);
                cfd[inf][1]  = m_ndcrd(ind, 1);
                cfd[0  ][1] += m_ndcrd(ind, 1);
                cfd[inf][2]  = m_ndcrd(ind, 2);
                cfd[0  ][2] += m_ndcrd(ind, 2);
            }
            cfd[nnd+1][0] = cfd[1][0];
            cfd[nnd+1][1] = cfd[1][1];
            cfd[nnd+1][2] = cfd[1][2];
            cfd[0][0] /= nnd;
            cfd[0][1] /= nnd;
            cfd[0][2] /= nnd;
            // calculate area.
            real_type voc = 0.0;
            m_fccnd(ifc, 0) = m_fccnd(ifc, 1) = m_fccnd(ifc, 2) = 0.0;
            for (size_t inf = 1 ; inf <= nnd ; ++inf)
            {
                crd[0] = (cfd[0][0] + cfd[inf][0] + cfd[inf+1][0])/3;
                crd[1] = (cfd[0][1] + cfd[inf][1] + cfd[inf+1][1])/3;
                crd[2] = (cfd[0][2] + cfd[inf][2] + cfd[inf+1][2])/3;
                real_type const du0 = cfd[inf][0] - cfd[0][0];
                real_type const du1 = cfd[inf][1] - cfd[0][1];
                real_type const du2 = cfd[inf][2] - cfd[0][2];
                real_type const dv0 = cfd[inf+1][0] - cfd[0][0];
                real_type const dv1 = cfd[inf+1][1] - cfd[0][1];
                real_type const dv2 = cfd[inf+1][2] - cfd[0][2];
                real_type const dw0 = du1*dv2 - du2*dv1;
                real_type const dw1 = du2*dv0 - du0*dv2;
                real_type const dw2 = du0*dv1 - du1*dv0;
                real_type vob = std::sqrt(dw0*dw0 + dw1*dw1 + dw2*dw2);
                m_fccnd(ifc, 0) += crd[0] * vob;
                m_fccnd(ifc, 1) += crd[1] * vob;
                m_fccnd(ifc, 2) += crd[2] * vob;
                voc += vob;
            }
            m_fccnd(ifc, 0) /= voc;
            m_fccnd(ifc, 1) /= voc;
            m_fccnd(ifc, 2) /= voc;
        }
    }

    // compute face normal vector and area.
    if (NDIM == 2)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // 2D faces are always lines.
            int_type const ind1 = m_fcnds(ifc, 1);
            int_type const ind2 = m_fcnds(ifc, 2);
            // face normal.
            m_fcnml(ifc, 0) = m_ndcrd(ind2, 1) - m_ndcrd(ind1, 1);
            m_fcnml(ifc, 1) = m_ndcrd(ind1, 0) - m_ndcrd(ind2, 0);
            // face ara.
            m_fcara(ifc) = std::sqrt(m_fcnml(ifc, 0)*m_fcnml(ifc, 0) + m_fcnml(ifc, 1)*m_fcnml(ifc, 1));
            // normalize face normal.
            m_fcnml(ifc, 0) /= m_fcara(ifc);
            m_fcnml(ifc, 1) /= m_fcara(ifc);
        }
    }
    else if (NDIM == 3)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // compute radial vector.
            std::array<std::array<real_type, NDIM>, FCMND> radvec; // NOLINT(cppcoreguidelines-pro-type-member-init)
            size_t const nnd = m_fcnds(ifc, 0);
            for (size_t inf = 0 ; inf < nnd ; ++inf)
            {
                int_type const ind = m_fcnds(ifc, inf+1);
                radvec[inf][0] = m_ndcrd(ind, 0) - m_fccnd(ifc, 0);
                radvec[inf][1] = m_ndcrd(ind, 1) - m_fccnd(ifc, 1);
                radvec[inf][2] = m_ndcrd(ind, 2) - m_fccnd(ifc, 2);
            }
            // compute cross product.
            m_fcnml(ifc, 0) = radvec[nnd-1][1]*radvec[0][2]
                            - radvec[nnd-1][2]*radvec[0][1];
            m_fcnml(ifc, 1) = radvec[nnd-1][2]*radvec[0][0]
                            - radvec[nnd-1][0]*radvec[0][2];
            m_fcnml(ifc, 2) = radvec[nnd-1][0]*radvec[0][1]
                            - radvec[nnd-1][1]*radvec[0][0];
            for (size_t ind = 1 ; ind < nnd ; ++ind)
            {
                m_fcnml(ifc, 0) += radvec[ind-1][1]*radvec[ind][2]
                                 - radvec[ind-1][2]*radvec[ind][1];
                m_fcnml(ifc, 1) += radvec[ind-1][2]*radvec[ind][0]
                                 - radvec[ind-1][0]*radvec[ind][2];
                m_fcnml(ifc, 2) += radvec[ind-1][0]*radvec[ind][1]
                                 - radvec[ind-1][1]*radvec[ind][0];
            }
            // compute face area.
            m_fcara(ifc) = std::sqrt
            (
                m_fcnml(ifc, 0)*m_fcnml(ifc, 0)
              + m_fcnml(ifc, 1)*m_fcnml(ifc, 1)
              + m_fcnml(ifc, 2)*m_fcnml(ifc, 2)
            );
            // normalize normal vector.
            m_fcnml(ifc, 0) /= m_fcara(ifc);
            m_fcnml(ifc, 1) /= m_fcara(ifc);
            m_fcnml(ifc, 2) /= m_fcara(ifc);
            // get real face area.
            m_fcara(ifc) /= 2.0;
        }
    }

    // compute cell centers.
    if (NDIM == 2)
    {
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (CellType::TRIANGLE == m_cltpn(icl)))
            {
                real_type voc = 0.0;
                {
                    int_type const ind = m_clnds(icl, 1);
                    real_type const vob = m_fcara(m_clfcs(icl, 2));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                }
                {
                    int_type const ind = m_clnds(icl, 2);
                    real_type const vob = m_fcara(m_clfcs(icl, 3));
                    voc += vob;
                    m_clcnd(icl, 0) += vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) += vob*m_ndcrd(ind, 1);
                }
                {
                    int_type const ind = m_clnds(icl, 3);
                    real_type const vob = m_fcara(m_clfcs(icl, 1));
                    voc += vob;
                    m_clcnd(icl, 0) += vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) += vob*m_ndcrd(ind, 1);
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
            }
            else // centroids.
            {
                // averaged point.
                std::array<real_type, NDIM> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
                crd[0] = crd[1] = 0.0;
                size_t const nnd = m_clnds(icl, 0);
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = m_clnds(icl, inc);
                    crd[0] += m_ndcrd(ind, 0);
                    crd[1] += m_ndcrd(ind, 1);
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                m_clcnd(icl, 0) = m_clcnd(icl, 1) = 0.0;
                size_t const nfc = m_clfcs(icl, 0);
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    int_type const ifc = m_clfcs(icl, ifl);
                    real_type const du0 = crd[0] - m_fccnd(ifc, 0);
                    real_type const du1 = crd[1] - m_fccnd(ifc, 1);
                    real_type vob = fabs(du0*m_fcnml(ifc, 0) + du1*m_fcnml(ifc, 1)) * m_fcara(ifc);
                    voc += vob;
                    real_type const dv0 = m_fccnd(ifc, 0) + du0/3;
                    real_type const dv1 = m_fccnd(ifc, 1) + du1/3;
                    m_clcnd(icl, 0) += dv0 * vob;
                    m_clcnd(icl, 1) += dv1 * vob;
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
            }
        }
    }
    else if (NDIM == 3)
    {
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (m_cltpn(icl) == 5))
            {
                real_type voc = 0.0;
                {
                    int_type const ind = m_clnds(icl, 1);
                    real_type const vob = m_fcara(m_clfcs(icl, 4));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                {
                    int_type const ind = m_clnds(icl, 2);
                    real_type const vob = m_fcara(m_clfcs(icl, 3));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                {
                    int_type const ind = m_clnds(icl, 3);
                    real_type const vob = m_fcara(m_clfcs(icl, 2));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                {
                    int_type const ind = m_clnds(icl, 4);
                    real_type const vob = m_fcara(m_clfcs(icl, 1));
                    voc += vob;
                    m_clcnd(icl, 0) = vob*m_ndcrd(ind, 0);
                    m_clcnd(icl, 1) = vob*m_ndcrd(ind, 1);
                    m_clcnd(icl, 2) = vob*m_ndcrd(ind, 2);
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
                m_clcnd(icl, 2) /= voc;
            }
            else // centroids.
            {
                // averaged point.
                std::array<real_type, NDIM> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
                crd[0] = crd[1] = crd[2] = 0.0;
                size_t const nnd = m_clnds(icl, 0);
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = m_clnds(icl, inc);
                    crd[0] += m_ndcrd(ind, 0);
                    crd[1] += m_ndcrd(ind, 1);
                    crd[2] += m_ndcrd(ind, 2);
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                crd[2] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                m_clcnd(icl, 0) = m_clcnd(icl, 1) = m_clcnd(icl, 2) = 0.0;
                size_t const nfc = m_clfcs(icl, 0);
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    int_type const ifc = m_clfcs(icl, ifl);
                    real_type const du0 = crd[0] - m_fccnd(ifc, 0);
                    real_type const du1 = crd[1] - m_fccnd(ifc, 1);
                    real_type const du2 = crd[2] - m_fccnd(ifc, 2);
                    real_type const vob = fabs(du0*m_fcnml(ifc, 0) + du1*m_fcnml(ifc, 1) + du2*m_fcnml(ifc, 2)) * m_fcara(ifc);
                    voc += vob;
                    real_type const dv0 = m_fccnd(ifc, 0) + du0/4;
                    real_type const dv1 = m_fccnd(ifc, 1) + du1/4;
                    real_type const dv2 = m_fccnd(ifc, 2) + du2/4;
                    m_clcnd(icl, 0) += dv0 * vob;
                    m_clcnd(icl, 1) += dv1 * vob;
                    m_clcnd(icl, 2) += dv2 * vob;
                }
                m_clcnd(icl, 0) /= voc;
                m_clcnd(icl, 1) /= voc;
                m_clcnd(icl, 2) /= voc;
            }
        }
    }

    // compute volume for each cell.
    auto reorder_face = [this](int_type ifc)
    {
        size_t const nnd = m_fcnds(ifc, 0);
        std::array<int_type, FCMND> ndstf; // NOLINT(cppcoreguidelines-pro-type-member-init)
        for (size_t jt = 0 ; jt < nnd ; ++jt)
        {
            ndstf[jt] = m_fcnds(ifc, nnd-jt);
        }
        for (size_t jt = 0 ; jt < nnd ; ++jt)
        {
            m_fcnds(ifc, jt+1) = ndstf[jt];
        }
        for (size_t idm = 0 ; idm < NDIM ; ++idm)
        {
            m_fcnml(ifc, idm) = -m_fcnml(ifc, idm);
        }
    };
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        m_clvol(icl) = 0.0;
        size_t const nfc = m_clfcs(icl, 0);
        for (size_t it = 1 ; it <= nfc ; ++it)
        {
            int_type const ifc = m_clfcs(icl, it);
            // calculate volume associated with each face.
            real_type vol = 0.0;
            for (size_t idm = 0 ; idm < NDIM ; ++idm)
            {
                vol += (m_fccnd(ifc, idm) - m_clcnd(icl, idm)) * m_fcnml(ifc, idm);
            }
            vol *= m_fcara(ifc);
            // check if need to reorder node definition and connecting cell
            // list for the face.
            size_t const this_fcl = m_fccls(ifc, 0);
            if (vol < 0.0)
            {
                if (this_fcl == icl) { reorder_face(ifc); }
                vol = -vol;
            }
            else
            {
                if (this_fcl != icl) { reorder_face(ifc); }
            }
            // accumulate the volume for the cell.
            m_clvol(icl) += vol;
        }
        // calculate the real volume.
        m_clvol(icl) /= NDIM;
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
