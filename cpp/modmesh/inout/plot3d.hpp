#pragma once

#include <queue>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <modmesh/base.hpp>
#include <modmesh/mesh/mesh.hpp>
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/inout/inout_util.hpp>

namespace modmesh
{

namespace inout
{

class Plot3d
    : public NumberBase<int32_t, double>
{
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:
    explicit Plot3d(const std::string & data);

    ~Plot3d() = default;

    Plot3d() = delete;
    Plot3d(Plot3d const & other) = delete;
    Plot3d(Plot3d && other) = delete;
    Plot3d & operator=(Plot3d const & other) = delete;
    Plot3d & operator=(Plot3d && other) = delete;

    std::shared_ptr<StaticMesh> to_block();
    void build_interior(const std::shared_ptr<StaticMesh> & blk);

private:
    inline uint_type coordinate_to_node_id(uint_type blk, uint_type x, uint_type y, uint_type z)
    {
        return blk > 0 ? m_blk_sizes(blk - 1) : 0 + x + y * m_x_shape(blk) + z * m_x_shape(blk) * m_y_shape(blk);
    }

    void parseCoordinates(const uint_type nblocks);
    void buildHexahedronElements(const uint_type nblocks);

    std::stringstream stream;

    SimpleArray<real_type> m_nds;
    SimpleArray<uint_type> m_x_shape;
    SimpleArray<uint_type> m_y_shape;
    SimpleArray<uint_type> m_z_shape;
    SimpleArray<uint_type> m_blk_sizes;

    std::unordered_map<uint_type, small_vector<uint_type>> m_elems;
};

} // namespace inout

} // namespace modmesh

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
