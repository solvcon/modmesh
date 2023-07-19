#include <modmesh/io/io.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

std::string g_test_file_path;

TEST(Gmsh_Parser, NonCellTypeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(0);
    EXPECT_EQ(ele_def.ndim(), 0);
    EXPECT_EQ(ele_def.nnds(), 0);
    EXPECT_EQ(ele_def.mmtpn(), 0);
    EXPECT_EQ(ele_def.mmcl().empty(), true);
}

TEST(Gmsh_Parser, Line2NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(1);
    EXPECT_EQ(ele_def.ndim(), 1);
    EXPECT_EQ(ele_def.nnds(), 2);
    EXPECT_EQ(ele_def.mmtpn(), 2);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1));
}

TEST(Gmsh_Parser, Triangle3NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(2);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 3);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, Quadrangle4NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(3);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 4);
    EXPECT_EQ(ele_def.mmtpn(), 3);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, Tetrahedron4NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(4);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 4);
    EXPECT_EQ(ele_def.mmtpn(), 6);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, Hexahedron8NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(5);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 8);
    EXPECT_EQ(ele_def.mmtpn(), 5);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(Gmsh_Parser, Prism6NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(6);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 6);
    EXPECT_EQ(ele_def.mmtpn(), 7);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 2, 1, 3, 5, 4));
}

TEST(Gmsh_Parser, Pryamid5NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(7);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 5);
    EXPECT_EQ(ele_def.mmtpn(), 8);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST(Gmsh_Parser, Line3NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(8);
    EXPECT_EQ(ele_def.ndim(), 1);
    EXPECT_EQ(ele_def.nnds(), 3);
    EXPECT_EQ(ele_def.mmtpn(), 2);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1));
}

TEST(Gmsh_Parser, Triangle6NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(9);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 6);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, Quadrangle9NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(10);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 9);
    EXPECT_EQ(ele_def.mmtpn(), 3);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, tetrahedron10NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(11);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 10);
    EXPECT_EQ(ele_def.mmtpn(), 6);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, hexahedron27NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(12);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 27);
    EXPECT_EQ(ele_def.mmtpn(), 5);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(Gmsh_Parser, Prism18NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(13);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 18);
    EXPECT_EQ(ele_def.mmtpn(), 7);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 2, 1, 3, 5, 4));
}

TEST(Gmsh_Parser, Pyramid14NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(14);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 14);
    EXPECT_EQ(ele_def.mmtpn(), 8);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST(Gmsh_Parser, Point1NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(15);
    EXPECT_EQ(ele_def.ndim(), 0);
    EXPECT_EQ(ele_def.nnds(), 1);
    EXPECT_EQ(ele_def.mmtpn(), 1);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0));
}

TEST(Gmsh_Parser, Quadrangle8NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(16);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 8);
    EXPECT_EQ(ele_def.mmtpn(), 3);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, hexahedron20NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(17);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 20);
    EXPECT_EQ(ele_def.mmtpn(), 5);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(Gmsh_Parser, Prism15NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(18);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 15);
    EXPECT_EQ(ele_def.mmtpn(), 7);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 2, 1, 3, 5, 4));
}

TEST(Gmsh_Parser, Pyramid13NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(19);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 13);
    EXPECT_EQ(ele_def.mmtpn(), 8);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST(Gmsh_Parser, IncompleteTriangle9NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(20);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 9);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, Triangle10NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(21);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 10);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, IncompleteTriangle12NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(22);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 12);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, Triangle15NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(23);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 15);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, IncompleteTriangle15NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(24);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 15);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, IncompleteTriangle21NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(25);
    EXPECT_EQ(ele_def.ndim(), 2);
    EXPECT_EQ(ele_def.nnds(), 21);
    EXPECT_EQ(ele_def.mmtpn(), 4);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2));
}

TEST(Gmsh_Parser, Edge4NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(26);
    EXPECT_EQ(ele_def.ndim(), 1);
    EXPECT_EQ(ele_def.nnds(), 4);
    EXPECT_EQ(ele_def.mmtpn(), 2);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1));
}

TEST(Gmsh_Parser, Edge5NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(27);
    EXPECT_EQ(ele_def.ndim(), 1);
    EXPECT_EQ(ele_def.nnds(), 5);
    EXPECT_EQ(ele_def.mmtpn(), 2);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1));
}

TEST(Gmsh_Parser, Edge6NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(28);
    EXPECT_EQ(ele_def.ndim(), 1);
    EXPECT_EQ(ele_def.nnds(), 6);
    EXPECT_EQ(ele_def.mmtpn(), 2);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1));
}

TEST(Gmsh_Parser, Tetrahedron20NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(29);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 20);
    EXPECT_EQ(ele_def.mmtpn(), 6);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, Tetrahedron35NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(30);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 35);
    EXPECT_EQ(ele_def.mmtpn(), 6);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, Tetrahedron56NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(31);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 56);
    EXPECT_EQ(ele_def.mmtpn(), 6);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3));
}

TEST(Gmsh_Parser, Hexahedron64NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(92);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 64);
    EXPECT_EQ(ele_def.mmtpn(), 5);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(Gmsh_Parser, Hexahedron125NodeDefinition)
{
    namespace gmsh = modmesh::IO::Gmsh;
    auto ele_def = gmsh::ElementDef::by_id(93);
    EXPECT_EQ(ele_def.ndim(), 3);
    EXPECT_EQ(ele_def.nnds(), 125);
    EXPECT_EQ(ele_def.mmtpn(), 5);
    EXPECT_THAT(ele_def.mmcl(), testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}
