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
void StaticMeshBase<D, ND>::build_boundary()
{
    assert(0 == m_nbound); // nothing should touch m_nbound beforehand.
    for (size_t it = 0 ; it < fccls().shape(0) ; ++it)
    {
        if (fccls()(it, 1) < 0)
        {
            m_nbound += 1;
        }
    }
    SimpleArray<int_type>(std::vector<size_t>{m_nbound, StaticMeshBC::BFREL}, -1).swap(m_bndfcs);

    std::vector<int_type> allfacn(m_nbound);
    size_t ait = 0;
    for (size_t ifc = 0 ; ifc < nface() ; ++ifc)
    {
        if (fcjcl(ifc) < 0)
        {
            assert(ait < allfacn.size());
            allfacn[ait] = ifc;
            ++ait;
        }
    }

    std::vector<bool> specified(m_nbound, false);
    size_t ibfc = 0;
    ssize_t nleft = m_nbound;
    for (size_t ibnd = 0 ; ibnd < m_bcs.size() ; ++ibnd)
    {
        StaticMeshBC & bnd = m_bcs[ibnd];
        auto & bfacn = bnd.facn();
        for (size_t bfit = 0 ; bfit < bfacn.nbody() ; ++bfit)
        {
            /**
             * First column is the face index in block.  The second column is the face
             * index in bndfcs.  The third column is the face index of the related
             * block (if exists).
             */
            m_bndfcs(ibfc, 0) = bfacn(bfit, 0);
            m_bndfcs(ibfc, 1) = ibnd;
            bfacn(bfit, 1) = static_cast<int_type>(ibfc);
            auto found = std::find(allfacn.begin(), allfacn.end(), bfacn(bfit, 0));
            if (allfacn.end() != found)
            {
                specified.at(found - allfacn.begin()) = true;
                --nleft;
            }
            ++ibfc;
        }
    }
    assert(nleft >= 0);

    if (nleft != 0)
    {
        StaticMeshBC bnd(static_cast<size_t>(nleft));
        auto & bfacn = bnd.facn();
        size_t bfit = 0;
        size_t ibnd = m_bcs.size();
        for (size_t sit = 0 ; sit < m_nbound ; ++sit)   // Specified ITerator.
        {
            if (!specified[sit])
            {
                m_bndfcs(ibfc, 0) = allfacn[sit];
                m_bndfcs(ibfc, 1) = ibnd;
                bfacn(bfit, 0) = allfacn[sit];
                bfacn(bfit, 1) = static_cast<int_type>(ibfc);
                ++ibfc;
                ++bfit;
            }
        }
        m_bcs.push_back(std::move(bnd));
        assert(m_bcs.size() == ibnd+1);
    }
    assert(ibfc == m_nbound);
}

template < typename D /* derived type */, uint8_t ND >
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
void StaticMeshBase<D, ND>::build_ghost()
{

    std::tie(m_ngstnode, m_ngstface, m_ngstcell) = count_ghost();

    #define MM_DECL_GHOST_SWAP1(N, T, D1, I) \
    { \
        SimpleArray<T> arr(std::vector<size_t>{m_ngst##D1 + m_n##D1}, I); \
        arr.set_nghost(m_ngst##D1); \
        for (int_type it = 0 ; it < static_cast<int_type>(m_n##D1) ; ++ it) \
        { \
            arr(it) = m_##N(it); \
        } \
        arr.swap(m_##N); \
    }

    #define MM_DECL_GHOST_SWAP2(N, T, D1, D2, I) \
    { \
        SimpleArray<T> arr(std::vector<size_t>{m_ngst##D1 + m_n##D1, D2}, I); \
        arr.set_nghost(m_ngst##D1); \
        for (int_type it = 0 ; it < static_cast<int_type>(m_n##D1) ; ++ it) \
        { \
            for (int_type jt = 0 ; jt < static_cast<int_type>(D2) ; ++jt) \
            { \
                arr(it, jt) = m_##N(it, jt); \
            } \
            arr(it) = m_##N(it); \
        } \
        arr.swap(m_##N); \
    }

    // geometry arrays.
    MM_DECL_GHOST_SWAP2(ndcrd, real_type, node, NDIM, 0)
    MM_DECL_GHOST_SWAP2(fccnd, real_type, face, NDIM, 0)
    MM_DECL_GHOST_SWAP2(fcnml, real_type, face, NDIM, 0)
    MM_DECL_GHOST_SWAP1(fcara, real_type, face, 0)
    MM_DECL_GHOST_SWAP2(clcnd, real_type, cell, NDIM, 0)
    MM_DECL_GHOST_SWAP1(clvol, real_type, cell, 0)
    // meta arrays.
    MM_DECL_GHOST_SWAP1(fctpn, int_type, face, 0)
    MM_DECL_GHOST_SWAP1(cltpn, int_type, cell, 0)
    MM_DECL_GHOST_SWAP1(clgrp, int_type, cell, -1)
    // connectivity arrays.
    MM_DECL_GHOST_SWAP2(fcnds, int_type, face, FCMND+1, -1)
    MM_DECL_GHOST_SWAP2(fccls, int_type, face, FCREL, -1)
    MM_DECL_GHOST_SWAP2(clnds, int_type, cell, CLMND+1, -1)
    MM_DECL_GHOST_SWAP2(clfcs, int_type, cell, CLMFC+1, -1)

    #undef MM_DECL_GHOST_SWAP1
    #undef MM_DECL_GHOST_SWAP2

    fill_ghost();

}

/**
 * @brief Count the number of ghost entities.
 *
 * @return std::tuple<size_t, size_t, size_t>
 *  ngstnode, ngstface, ngstcell
 */
template < typename D /* derived type */, uint8_t ND >
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
std::tuple<size_t, size_t, size_t> StaticMeshBase<D, ND>::count_ghost() const
{
    size_t ngstface = 0;
    size_t ngstnode = 0;
    for (size_t ibfc = 0 ; ibfc < m_nbound ; ++ibfc)
    {
        const int_type ifc = m_bndfcs(ibfc, 0);
        const int_type icl = m_fccls(ifc, 0);
        ngstface += CellType::by_id(m_cltpn(icl)).nface() - 1;
        ngstnode += m_clnds(icl, 0) - m_fcnds(ifc, 0);
    }
    return std::make_tuple(ngstnode, ngstface, m_nbound);
}

/**
 * Build all information for ghost cells by mirroring information from interior
 * cells.  The action includes:
 *
 * 1. define indices and build connectivities for ghost nodes, faces,
 *    and cells.  In the same loop, mirror the coordinates of interior
 *    nodes to ghost nodes.
 * 2. compute center coordinates for faces for ghost cells.
 * 3. compute normal vectors and areas for faces for ghost cells.
 * 4. compute center coordinates for ghost cells.
 * 5. compute volume for ghost cells.
 *
 * NOTE: all the metric, type and connnectivities data passed in this
 * subroutine are SHARED arrays rather than interior arrays.  The
 * indices for ghost information should be carefully treated.  All the
 * ghost indices are negative in shared arrays.
 */
template < typename D /* derived type */, uint8_t ND >
/* NOLINTNEXTLINE(readability-function-cognitive-complexity) */
void StaticMeshBase<D, ND>::fill_ghost()
{
    // arrays.
    std::array<std::array<real_type, NDIM>, FCMND+2> cfd; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<real_type, NDIM> crd; // NOLINT(cppcoreguidelines-pro-type-member-init)
    std::array<std::array<real_type, NDIM>, FCMND> radvec; // NOLINT(cppcoreguidelines-pro-type-member-init)

    std::vector<int_type> gstndmap(nnode(), nnode());

    // create ghost entities and buil connectivities and by the way mirror node
    // coordinate.
    int_type ignd = -1;
    int_type igfc = -1;
    for (int_type igcl = -1 ; igcl >= -static_cast<int_type>(ngstcell()) ; --igcl)
    {
        int_type const ibfc = m_bndfcs(-igcl-1, 0);
        int_type const icl = m_fccls(ibfc, 0);
        // copy cell type and group.
        m_cltpn(igcl) = m_cltpn(icl);
        m_clgrp(igcl) = m_clgrp(icl);
        // process node list in ghost cell.
        for (size_t inl = 0 ; inl <= CLMND ; ++inl) // copy nodes from current in-cell.
        {
            m_clnds(igcl, inl) = m_clnds(icl, inl);
        }
        for (size_t inl = 1 ; inl <= static_cast<size_t>(m_clnds(icl, 0)) ; ++inl)
        {
            int_type const ind = m_clnds(icl, inl);
            // try to find the node in the boundary face.
            bool mk_found = false;
            for (size_t inf = 1 ; inf <= static_cast<size_t>(m_fcnds(ibfc, 0)) ; ++inf)
            {
                if (ind == m_fcnds(ibfc, inf))
                {
                    mk_found = true;
                    break;
                }
            }
            // if not found, it should be a ghost node.
            if (!mk_found)
            {
                gstndmap[ind] = ignd; // record map for face processing.
                m_clnds(igcl, inl) = ignd; // save to clnds.
                // mirror coordinate of ghost cell.
                // NOTE: fcnml always points outward.
                real_type dist = 0.0;
                for (size_t idm = 0 ; idm < NDIM ; ++idm)
                {
                    dist += (m_fccnd(ibfc, idm) - m_ndcrd(ind, idm)) * m_fcnml(ibfc, idm);
                }
                for (size_t idm = 0 ; idm < NDIM ; ++idm)
                {
                    m_ndcrd(igcl, idm) = m_ndcrd(icl, idm) + 2*dist*m_fcnml(ibfc, idm);
                }
                // decrement ghost node counter.
                ignd -= 1;
            }
        }
        // set the relating cell as ghost cell.
        m_fccls(ibfc, 1) = igcl;
        // process face list in ghost cell.
        for (size_t ifl = 0 ; ifl <= CLMFC ; ++ifl)
        {
            m_clfcs(igcl, ifl) = m_clfcs(icl, ifl); // copy in-face to ghost.
        }
        for (size_t ifl = 1 ; ifl <= static_cast<size_t>(m_clfcs(icl, 0)) ; ++ifl)
        {
            int_type const ifc = m_clfcs(icl, ifl);  // the face to be processed.
            if (ifc == ibfc) { continue; }  // if boundary face then skip.
            m_fctpn(igfc) = m_fctpn(ifc);  // copy face type.
            m_fccls(igfc, 0) = igcl;  // save to ghost fccls.
            m_clfcs(igcl, ifl) = igfc;  // save to ghost clfcs.
            // face-to-node connectivity.
            for (size_t inf = 0 ; inf <= FCMND ; ++inf)
            {
                m_fcnds(igfc, inf) = m_fcnds(ifc, inf);
            }
            for (size_t inf = 1 ; inf <= static_cast<size_t>(m_fcnds(igfc, 0)) ; ++inf)
            {
                int_type const ind = m_fcnds(igfc, inf);
                if (gstndmap[ind] != static_cast<int_type>(nnode()))
                {
                    m_fcnds(igfc, inf) = gstndmap[ind];  // save gstnode to fcnds.
                }
            }
            // decrement ghost face counter.
            igfc -= 1;
        }
        // erase node map record.
        for (size_t inl = 1 ; inl <= static_cast<size_t>(m_clnds(icl, 0)); ++inl)
        {
            gstndmap[m_clnds(icl, inl)] = nnode();
        }
    }

    // compute ghost face centroids.
    if (NDIM == 2)
    {
        // 2D faces must be edge.
        for (int_type ifc = -1 ; ifc >= -static_cast<int_type>(ngstface()) ; --ifc)
        {
            // point 1.
            int_type ind = m_fcnds(ifc, 1);
            m_fccnd(ifc, 0) = m_ndcrd(ind, 0);
            m_fccnd(ifc, 1) = m_ndcrd(ind, 1);
            // point 2.
            ind = m_fcnds(ifc, 2);
            m_fccnd(ifc, 0) += m_ndcrd(ind, 0);
            m_fccnd(ifc, 1) += m_ndcrd(ind, 1);
            // average.
            m_fccnd(ifc, 0) /= 2;
            m_fccnd(ifc, 1) /= 2;
        }
    }
    else if (NDIM == 3)
    {
        for (int_type ifc = -1 ; ifc >= -static_cast<int_type>(ngstface()) ; --ifc)
        {
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
            m_fccnd(ifc, 0) = m_fccnd(ifc, 1) = m_fccnd(ifc, 2) = 0.0;
            real_type voc = 0.0;
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
                real_type const vob = std::sqrt(dw0*dw0 + dw1*dw1 + dw2*dw2);
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

    // compute ghost face normal vector and area.
    if (NDIM == 2)
    {
        for (int_type ifc = -1 ; ifc >= -static_cast<int_type>(ngstface()) ; --ifc)
        {
            // 2D faces are always lines.
            int_type const ind = m_fcnds(ifc, 1);
            int_type const jnd = m_fcnds(ifc, 2);
            // face normal.
            m_fcnml(ifc, 0) = m_ndcrd(jnd, 1) - m_ndcrd(ind, 1);
            m_fcnml(ifc, 1) = m_ndcrd(ind, 0) - m_ndcrd(jnd, 0);
            // face ara.
            m_fcara(ifc) = std::sqrt(m_fcnml(ifc, 0)*m_fcnml(ifc, 0) + m_fcnml(ifc, 1)*m_fcnml(ifc, 1));
            // normalize face normal.
            m_fcnml(ifc, 0) /= m_fcara(ifc);
            m_fcnml(ifc, 1) /= m_fcara(ifc);
        }
    }
    else if (NDIM == 3)
    {
        for (int_type ifc = -1 ; ifc >= -static_cast<int_type>(ngstface()) ; --ifc)
        {
            // compute radial vector.
            size_t const nnd = m_fcnds(ifc);
            for (size_t inf = 0 ; inf < nnd ; ++inf)
            {
                int_type const ind = m_fcnds(ifc, inf+1);
                radvec[inf][0] = m_ndcrd(ind, 0) - m_fccnd(ifc, 0);
                radvec[inf][1] = m_ndcrd(ind, 1) - m_fccnd(ifc, 1);
                radvec[inf][2] = m_ndcrd(ind, 2) - m_fccnd(ifc, 2);
            }
            // compute cross product.
            m_fcnml(ifc, 0) = radvec[nnd-1][1]*radvec[0][2] - radvec[nnd-1][2]*radvec[0][1];
            m_fcnml(ifc, 1) = radvec[nnd-1][2]*radvec[0][0] - radvec[nnd-1][0]*radvec[0][2];
            m_fcnml(ifc, 2) = radvec[nnd-1][0]*radvec[0][1] - radvec[nnd-1][1]*radvec[0][0];
            for (size_t ind = 1 ; ind < nnd ; ++ind)
            {
                m_fcnml(ifc, 0) += radvec[ind-1][1]*radvec[ind][2] - radvec[ind-1][2]*radvec[ind][1];
                m_fcnml(ifc, 1) += radvec[ind-1][2]*radvec[ind][0] - radvec[ind-1][0]*radvec[ind][2];
                m_fcnml(ifc, 2) += radvec[ind-1][0]*radvec[ind][1] - radvec[ind-1][1]*radvec[ind][0];
            }
            // compute face area.
            m_fcara(ifc, 0) = std::sqrt
            (
                m_fcnml(ifc, 0)*m_fcnml(ifc, 0)
              + m_fcnml(ifc, 1)*m_fcnml(ifc, 1)
              + m_fcnml(ifc, 2)*m_fcnml(ifc, 2)
            );
            // normalize normal vector.
            m_fcnml(ifc, 0) /= m_fcnml(ifc);
            m_fcnml(ifc, 1) /= m_fcnml(ifc);
            m_fcnml(ifc, 2) /= m_fcnml(ifc);
            // get real face area.
            m_fcnml(ifc) /= 2.0;
        }
    }

    // compute cell centroids.
    if (NDIM == 2)
    {
        for (int_type icl = -1 ; icl >= -static_cast<int_type>(ngstcell()) ; --icl)
        {
            // averaged point.
            crd[0] = crd[1] = 0.0;
            size_t const nnd = m_clnds(icl, 0);
            for (size_t inl = 1 ; inl <= nnd ; ++inl)
            {
                int_type const ind = m_clnds(icl, inl);
                crd[0] += m_ndcrd(ind, 0);
                crd[1] += m_ndcrd(ind, 1);
            }
            crd[0] /= nnd;
            crd[1] /= nnd;
            // weight centroid.
            m_clcnd(icl, 0) = m_clcnd(icl, 1) = 0.0;
            real_type voc = 0.0;
            size_t nfc = m_clfcs(icl, 0);
            for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
            {
                int_type const ifc = m_clfcs(icl, ifl);
                real_type const du0 = crd[0] - m_fccnd(ifc, 0);
                real_type const du1 = crd[1] - m_fccnd(ifc, 1);
                real_type const vob = std::abs(du0*m_fcnml(ifc, 0) + du1*m_fcnml(ifc, 1)) * m_fcara(ifc);
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
    else if (NDIM == 3)
    {
        for (int_type icl = -1 ; icl >= -static_cast<int_type>(ngstcell()) ; --icl)
        {
            // averaged point.
            crd[0] = crd[1] = crd[2] = 0.0;
            size_t const nnd = m_clnds(icl, 0);
            for (size_t inl = 1 ; inl <= nnd ; ++inl)
            {
                int_type const ind = m_clnds(icl, inl);
                crd[0] += m_ndcrd(ind, 0);
                crd[1] += m_ndcrd(ind, 1);
                crd[2] += m_ndcrd(ind, 2);
            }
            crd[0] /= nnd;
            crd[1] /= nnd;
            crd[2] /= nnd;
            // weight centroid.
            m_clcnd(icl, 0) = m_clcnd(icl, 1) = m_clcnd(icl, 2) = 0.0;
            real_type voc = 0.0;
            size_t const nfc = m_clfcs(icl, 0);
            for (size_t ifl = 1 ; ifl <= nfc ; ++ifl)
            {
                int_type const ifc = m_clfcs(icl, ifl);
                real_type const du0 = crd[0] - m_fccnd(ifc, 0);
                real_type const du1 = crd[1] - m_fccnd(ifc, 1);
                real_type const du2 = crd[2] - m_fccnd(ifc, 2);
                real_type const vob = std::fabs
                (
                    (du0*m_fcnml(ifc, 0) + du1*m_fcnml(ifc, 1) + du2*m_fcnml(ifc, 2))
                  * m_fcara(ifc)
                );
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

    // compute volume for each ghost cell.
    for (int_type icl = -1 ; icl >= -static_cast<int_type>(ngstcell()) ; --icl)
    {
        m_clvol(icl) = 0.0;
        for (size_t it = 1 ; it <= static_cast<size_t>(m_clfcs(icl, 0)) ; ++it)
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
            if (vol < 0.0)
            {
                if (m_fccls(ifc, 0) == icl)
                {
                    for (size_t idm = 0 ; idm < NDIM ; ++idm)
                    {
                        m_fcnml(ifc, idm) = -m_fcnml(ifc, idm);
                    }
                }
                vol = -vol;
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
