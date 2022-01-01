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
    int_type computed_nface = -1;

    // create temporary tables.
    SimpleArray<int_type> tclfcs(small_vector<size_t>{ncell(), CLMFC+1}, -1);
    SimpleArray<int_type> tfctpn(small_vector<size_t>{mface}, -1);
    SimpleArray<int_type> tfcnds(small_vector<size_t>{mface, FCMND+1}, -1);
    SimpleArray<int_type> tfccls(small_vector<size_t>{mface, FCNCL}, -1);
    int_type * lclfcs = tclfcs.body();
    int_type * lfctpn = tfctpn.body();
    int_type * lfcnds = tfcnds.body();
    int_type * lfccls = tfccls.body();

    // extract face definition from the node list of cells.
    int_type * pcltpn = cltpn().body();
    int_type * pclnds = clnds().body();
    int_type * pclfcs = lclfcs;
    int_type * pfctpn = lfctpn;
    int_type * pfcnds = lfcnds;
    int_type ifc = 0;
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        int_type const tpnicl = pcltpn[0];
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
        pcltpn += 1;
        pclnds += CLMND+1;
        pclfcs += CLMFC+1;
    };

    // build the hash table, to know what faces connect to each node.
    /// first pass: get the maximum number of faces.
    std::vector<int_type> ndnfc(nnode());
    for (size_t ind = 0 ; ind < nnode() ; ++ind) // initialize.
    {
        ndnfc[ind] = 0;
    }
    pfcnds = lfcnds; // count.
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type inf = 1 ; inf <= pfcnds[0] ; ++inf)
        {
            int_type const ind = pfcnds[inf];  // node of interest.
            ndnfc[ind] += 1;    // increment counting.
        };
        // advance pointers.
        pfcnds += FCMND+1;
    }
    int_type ndmfc = 0;  // get maximum.
    for (size_t ind = 0 ; ind < nnode() ; ++ind)
    {
        if (ndnfc[ind] > ndmfc)
        {
            ndmfc = ndnfc[ind];
        }
    }
    assert(ndmfc >= 0); // FIXME: Throw an exception.
    /// second pass: scan again to build hash table.
    SimpleArray<int_type> ndfcs(small_vector<size_t>{nnode(), static_cast<size_t>(ndmfc)+1});
    for (size_t ind = 0 ; ind < nnode() ; ++ind)
    {
        ndfcs(ind, 0) = 0;
        for (int_type it = 1 ; it <= ndmfc ; ++it)
        {
            ndfcs(ind, it) = -1;
        }
    }
    pfcnds = lfcnds; // build hash table mapping from node to face.
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type inf = 1; inf <= pfcnds[0]; ++inf)
        {
            int_type const ind = pfcnds[inf];  // node of interest.
            ndfcs(ind, 0) += 1;
            ndfcs(ind, ndfcs(ind, 0)) = ifc;
        }
        // advance pointers.
        pfcnds += FCMND+1;
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
            int_type * pifcnds = lfcnds + ifc*(FCMND+1);
            int_type const nd1 = pifcnds[1];    // take only the FIRST node of a face.
            for (int_type it = 1 ; it <= ndfcs(nd1, 0) ; ++it)
            {
                size_t const jfc = ndfcs(nd1, it);
                // test for duplication.
                if ((jfc != ifc) && (lfctpn[jfc] == lfctpn[ifc]))
                {
                    int_type * pjfcnds = lfcnds + jfc*(FCMND+1);
                    int_type cond = pjfcnds[0];
                    // scan all nodes in ifc and jfc to see if all the same.
                    for (int_type jnf = 1 ; jnf <= pjfcnds[0] ; ++jnf)
                    {
                        for (int_type inf = 1 ; inf <= pifcnds[0] ; ++inf)
                        {
                            if (pjfcnds[jnf] == pifcnds[inf])
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
    int_type * pifcnds = lfcnds;
    int_type * pjfcnds = lfcnds;
    int_type * pifctpn = lfctpn;
    int_type * pjfctpn = lfctpn;
    int_type jfc = 0;
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        if (map[ifc] == ifc)
        {
            for (int_type inf = 0 ; inf <= FCMND ; ++inf)
            {
                pjfcnds[inf] = pifcnds[inf];
            }
            pjfctpn[0] = pifctpn[0];
            map2[ifc] = jfc;
            // increment j-face.
            jfc += 1;
            pjfcnds += FCMND+1;
            pjfctpn += 1;
        }
        else
        {
            map2[ifc] = map2[map[ifc]];
        }
        // advance pointers;
        pifcnds += FCMND+1;
        pifctpn += 1;
    }
    computed_nface = jfc;    // record deduplicated number of face.

    // rebuild faces in cells and build face neighboring, according to the
    // renewed face map.
    int_type * pfccls = lfccls; // initialize.
    for (size_t ifc = 0 ; ifc < mface ; ++ifc)
    {
        for (int_type it = 0 ; it < FCREL ; ++it)
        {
            pfccls[it] = -1;
        }
        // advance pointers;
        pfccls += FCREL;
    }
    pclfcs = lclfcs;
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        for (int_type ifl = 1 ; ifl <= pclfcs[0] ; ++ifl)
        {
            int_type const ifc = pclfcs[ifl];
            int_type const jfc = map2[ifc];
            // rebuild faces in cells.
            pclfcs[ifl] = jfc;
            // build face neighboring.
            pfccls = lfccls + jfc*FCREL;
            if (pfccls[0] == -1)
            {
                pfccls[0] = icl;
            }
            else if (pfccls[1] == -1)
            {
                pfccls[1] = icl;
            }
        }
        // advance pointers;
        pclfcs += CLMFC+1;
    };

    // recreate member tables.
    m_nface = computed_nface;
    m_fctpn.swap(SimpleArray<int_type>(small_vector<size_t>{nface()}));
    m_fcnds.swap(SimpleArray<int_type>(small_vector<size_t>{nface(), FCMND+1}));
    m_fccls.swap(SimpleArray<int_type>(small_vector<size_t>{nface(), FCNCL}));
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        for (size_t it = 0 ; it < clfcs().shape(1) ; ++it)
        {
            clfcs()(icl, it) = tclfcs(icl, it);
        }
    }
    for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
    {
        fctpn()(ifc) = tfctpn(ifc);
        for (size_t it = 0 ; it < fcnds().shape(1) ; ++it)
        {
            fcnds()(ifc, it) = tfcnds(ifc, it);
        }
        for (size_t it = 0 ; it < fccls().shape(1) ; ++it)
        {
            fccls()(ifc, it) = tfccls(ifc, it);
        }
    }
    m_fccnd.swap(SimpleArray<real_type>(small_vector<size_t>{nface(), ND}, 0));
    m_fcnml.swap(SimpleArray<real_type>(small_vector<size_t>{nface(), ND}, 0));
    m_fcara.swap(SimpleArray<real_type>(small_vector<size_t>{nface()}, 0));
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
    // arrays.
    std::array<int_type, FCMND> ndstf; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<std::array<real_type, NDIM>, FCMND+2> cfd; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<real_type, NDIM> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<std::array<real_type, NDIM>, FCMND> radvec; // NOLINT(cppcoreguidelines-pro-type-member-init)

    // utilized arrays.
    real_type * lndcrd = ndcrd().body();
    int_type  * lfcnds = fcnds().body();
    int_type  * lfccls = fccls().body();
    real_type * lfccnd = fccnd().body();
    real_type * lfcnml = fcnml().body();
    real_type * lfcara = fcara().body();
    int_type  * lcltpn = cltpn().body();
    int_type  * lclnds = clnds().body();
    int_type  * lclfcs = clfcs().body();
    real_type * lclcnd = clcnd().body();
    real_type * lclvol = clvol().body();

    // compute face centroids.
    int_type * pfcnds = lfcnds;
    real_type * pfccnd = lfccnd;
    if (NDIM == 2)
    {
        // 2D faces must be edge.
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // point 1.
            int_type ind = pfcnds[1];
            real_type const * pndcrd = lndcrd + ind*NDIM;
            pfccnd[0] = pndcrd[0];
            pfccnd[1] = pndcrd[1];
            // point 2.
            ind = pfcnds[2];
            pndcrd = lndcrd + ind*NDIM;
            pfccnd[0] += pndcrd[0];
            pfccnd[1] += pndcrd[1];
            // average.
            pfccnd[0] /= 2;
            pfccnd[1] /= 2;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += NDIM;
        }
    }
    else if (NDIM == 3)
    {
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // find averaged point.
            cfd[0][0] = cfd[0][1] = cfd[0][2] = 0.0;
            size_t const nnd = pfcnds[0];
            for (size_t inf = 1 ; inf <= nnd ; ++inf)
            {
                int_type ind = pfcnds[inf];
                real_type const * pndcrd = lndcrd + ind*NDIM;
                cfd[inf][0]  = pndcrd[0];
                cfd[0  ][0] += pndcrd[0];
                cfd[inf][1]  = pndcrd[1];
                cfd[0  ][1] += pndcrd[1];
                cfd[inf][2]  = pndcrd[2];
                cfd[0  ][2] += pndcrd[2];
            }
            cfd[nnd+1][0] = cfd[1][0];
            cfd[nnd+1][1] = cfd[1][1];
            cfd[nnd+1][2] = cfd[1][2];
            cfd[0][0] /= nnd;
            cfd[0][1] /= nnd;
            cfd[0][2] /= nnd;
            // calculate area.
            real_type voc = 0.0;
            pfccnd[0] = pfccnd[1] = pfccnd[2] = 0.0;
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
                real_type vob = sqrt(dw0*dw0 + dw1*dw1 + dw2*dw2);
                pfccnd[0] += crd[0] * vob;
                pfccnd[1] += crd[1] * vob;
                pfccnd[2] += crd[2] * vob;
                voc += vob;
            }
            pfccnd[0] /= voc;
            pfccnd[1] /= voc;
            pfccnd[2] /= voc;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += NDIM;
        }
    }

    // compute face normal vector and area.
    pfcnds = lfcnds;
    pfccnd = lfccnd;
    if (NDIM == 2)
    {
        real_type * pfcnml = lfcnml;
        real_type * pfcara = lfcara;
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // 2D faces are always lines.
            real_type const * pndcrd = lndcrd + pfcnds[1]*NDIM;
            real_type const * p2ndcrd = lndcrd + pfcnds[2]*NDIM;
            // face normal.
            pfcnml[0] = p2ndcrd[1] - pndcrd[1];
            pfcnml[1] = -(p2ndcrd[0] - pndcrd[0]);
            // face ara.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]);
            // normalize face normal.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
            // advance pointers.
            pfcnds += FCMND+1;
            pfcnml += NDIM;
            pfcara += 1;
        }
    }
    else if (NDIM == 3)
    {
        real_type * pfcnml = lfcnml;
        real_type * pfcara = lfcara;
        for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
        {
            // compute radial vector.
            size_t const nnd = pfcnds[0];
            for (size_t inf = 0 ; inf < nnd ; ++inf)
            {
                int_type ind = pfcnds[inf+1];
                real_type const * pndcrd = lndcrd + ind*NDIM;
                radvec[inf][0] = pndcrd[0] - pfccnd[0];
                radvec[inf][1] = pndcrd[1] - pfccnd[1];
                radvec[inf][2] = pndcrd[2] - pfccnd[2];
            }
            // compute cross product.
            pfcnml[0] = radvec[nnd-1][1]*radvec[0][2]
                      - radvec[nnd-1][2]*radvec[0][1];
            pfcnml[1] = radvec[nnd-1][2]*radvec[0][0]
                      - radvec[nnd-1][0]*radvec[0][2];
            pfcnml[2] = radvec[nnd-1][0]*radvec[0][1]
                      - radvec[nnd-1][1]*radvec[0][0];
            for (size_t ind = 1 ; ind < nnd ; ++ind)
            {
                pfcnml[0] += radvec[ind-1][1]*radvec[ind][2]
                           - radvec[ind-1][2]*radvec[ind][1];
                pfcnml[1] += radvec[ind-1][2]*radvec[ind][0]
                           - radvec[ind-1][0]*radvec[ind][2];
                pfcnml[2] += radvec[ind-1][0]*radvec[ind][1]
                           - radvec[ind-1][1]*radvec[ind][0];
            }
            // compute face area.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]
                           + pfcnml[2]*pfcnml[2]);
            // normalize normal vector.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
            pfcnml[2] /= pfcara[0];
            // get real face area.
            pfcara[0] /= 2.0;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += NDIM;
            pfcnml += NDIM;
            pfcara += 1;
        }
    }

    // compute cell centroids.
    int_type * pclnds = lclnds;
    int_type * pclfcs = lclfcs;
    if (NDIM == 2)
    {
        real_type * pclcnd = lclcnd;
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (lcltpn[icl] == 3))
            {
                real_type const * pndcrd = lndcrd + pclnds[1]*NDIM;
                real_type vob = lfcara[pclfcs[2]];
                real_type voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pndcrd = lndcrd + pclnds[2]*NDIM;
                vob = lfcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pndcrd = lndcrd + pclnds[3]*NDIM;
                vob = lfcara[pclfcs[1]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
            }
            else
            {
                // averaged point.
                crd[0] = crd[1] = 0.0;
                size_t const nnd = pclnds[0];
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = pclnds[inc];
                    real_type const * pndcrd = lndcrd + ind*NDIM;
                    crd[0] += pndcrd[0];
                    crd[1] += pndcrd[1];
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                pclcnd[0] = pclcnd[1] = 0.0;
                size_t const nfc = pclfcs[0];
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    int_type ifc = pclfcs[ifl];
                    pfccnd = lfccnd + ifc*NDIM;
                    real_type const * pfcnml = lfcnml + ifc*NDIM;
                    real_type const * pfcara = lfcara + ifc;
                    real_type const du0 = crd[0] - pfccnd[0];
                    real_type const du1 = crd[1] - pfccnd[1];
                    real_type vob = fabs(du0*pfcnml[0] + du1*pfcnml[1]) * pfcara[0];
                    voc += vob;
                    real_type const dv0 = pfccnd[0] + du0/3;
                    real_type const dv1 = pfccnd[1] + du1/3;
                    pclcnd[0] += dv0 * vob;
                    pclcnd[1] += dv1 * vob;
                }
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
            }
            // advance pointers.
            pclnds += CLMND+1;
            pclfcs += CLMFC+1;
            pclcnd += NDIM;
        }
    }
    else if (NDIM == 3)
    {
        real_type * pclcnd = lclcnd;
        for (size_t icl = 0 ; icl < ncell() ; ++icl)
        {
            if ((use_incenter()) && (lcltpn[icl] == 5))
            {
                real_type const * pndcrd = lndcrd + pclnds[1]*NDIM;
                real_type vob = lfcara[pclfcs[4]];
                real_type voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pclcnd[2] = vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[2]*NDIM;
                vob = lfcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[3]*NDIM;
                vob = lfcara[pclfcs[2]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[4]*NDIM;
                vob = lfcara[pclfcs[1]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
                pclcnd[2] /= voc;
            }
            else
            {
                // averaged point.
                crd[0] = crd[1] = crd[2] = 0.0;
                size_t const nnd = pclnds[0];
                for (size_t inc = 1 ; inc <= nnd ; ++inc)
                {
                    int_type const ind = pclnds[inc];
                    real_type const * pndcrd = lndcrd + ind*NDIM;
                    crd[0] += pndcrd[0];
                    crd[1] += pndcrd[1];
                    crd[2] += pndcrd[2];
                }
                crd[0] /= nnd;
                crd[1] /= nnd;
                crd[2] /= nnd;
                // weight centroid.
                real_type voc = 0.0;
                pclcnd[0] = pclcnd[1] = pclcnd[2] = 0.0;
                size_t const nfc = pclfcs[0];
                for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
                {
                    int_type ifc = pclfcs[ifl];
                    pfccnd = lfccnd + ifc*NDIM;
                    real_type const * pfcnml = lfcnml + ifc*NDIM;
                    real_type const * pfcara = lfcara + ifc;
                    real_type const du0 = crd[0] - pfccnd[0];
                    real_type const du1 = crd[1] - pfccnd[1];
                    real_type const du2 = crd[2] - pfccnd[2];
                    real_type vob = fabs(du0*pfcnml[0] + du1*pfcnml[1] + du2*pfcnml[2]) * pfcara[0];
                    voc += vob;
                    real_type const dv0 = pfccnd[0] + du0/4;
                    real_type const dv1 = pfccnd[1] + du1/4;
                    real_type const dv2 = pfccnd[2] + du2/4;
                    pclcnd[0] += dv0 * vob;
                    pclcnd[1] += dv1 * vob;
                    pclcnd[2] += dv2 * vob;
                }
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
                pclcnd[2] /= voc;
            }
            // advance pointers.
            pclnds += CLMND+1;
            pclfcs += CLMFC+1;
            pclcnd += NDIM;
        }
    }

    // compute volume for each cell.
    pclfcs = lclfcs;
    real_type const * pclcnd = lclcnd;
    real_type * pclvol = lclvol;
    for (size_t icl = 0 ; icl < ncell() ; ++icl)
    {
        pclvol[0] = 0.0;
        size_t const nfc = pclfcs[0];
        for (size_t it = 1 ; it <= nfc ; ++it)
        {
            int_type ifc = pclfcs[it];
            int_type * pfccls = lfccls + ifc*FCREL;
            pfcnds = lfcnds + ifc*(FCMND+1);
            pfccnd = lfccnd + ifc*NDIM;
            real_type * pfcnml = lfcnml + ifc*NDIM;
            real_type const * pfcara = lfcara + ifc;
            // calculate volume associated with each face.
            real_type vol = 0.0;
            for (size_t idm = 0 ; idm < NDIM ; ++idm)
            {
                vol += (pfccnd[idm] - pclcnd[idm]) * pfcnml[idm];
            }
            vol *= pfcara[0];
            // check if need to reorder node definition and connecting cell
            // list for the face.
            size_t const this_fcl = pfccls[0];
            if (vol < 0.0)
            {
                if (this_fcl == icl)
                {
                    size_t const nnd = pfcnds[0];
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        ndstf[jt] = pfcnds[nnd-jt];
                    }
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        pfcnds[jt+1] = ndstf[jt];
                    }
                    for (size_t idm = 0 ; idm < NDIM ; ++idm)
                    {
                        pfcnml[idm] = -pfcnml[idm];
                    }
                }
                vol = -vol;
            }
            else
            {
                if (this_fcl != icl)
                {
                    size_t const nnd = pfcnds[0];
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        ndstf[jt] = pfcnds[nnd-jt];
                    }
                    for (size_t jt = 0 ; jt < nnd ; ++jt)
                    {
                        pfcnds[jt+1] = ndstf[jt];
                    }
                    for (size_t idm = 0 ; idm < NDIM ; ++idm)
                    {
                        pfcnml[idm] = -pfcnml[idm];
                    }
                }
            }
            // accumulate the volume for the cell.
            pclvol[0] += vol;
        }
        // calculate the real volume.
        pclvol[0] /= NDIM;
        // advance pointers.
        pclfcs += CLMFC+1;
        pclcnd += NDIM;
        pclvol += 1;
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
