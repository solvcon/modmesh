#include <modmesh/toggle/RadixTree.hpp>
#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

template <typename T>
std::string get_info(modmesh::RadixTreeNode<T> & r)
{
    std::ostringstream oss;
    if (nullptr != r.get_prev())
    {
        oss << r.name() << " - " << r.data();
    }
    return oss.str();
}

TEST(RadixTree, construction)
{
    namespace mm = modmesh;
    mm::RadixTree<mm::TimedEntry> radix_tree;
    EXPECT_EQ(radix_tree.get_unique_node(), 0);
}

TEST(RadixTree, single_insertion)
{
    namespace mm = modmesh;
    mm::RadixTree<mm::TimedEntry> radix_tree;
    mm::TimedEntry & entry1 = radix_tree.entry("a");
    entry1.add_time(5.2);

    EXPECT_EQ(entry1.count(), 1);
    EXPECT_DOUBLE_EQ(entry1.time(), 5.2);
    EXPECT_EQ(radix_tree.get_unique_node(), 1);

    mm::RadixTreeNode<mm::TimedEntry> * node = radix_tree.get_current_node();
    EXPECT_EQ(node->name(), "a");
    EXPECT_EQ(get_info(*node), "a - Count: 1 - Time: 5.2");
}

TEST(RadixTree, multiple_insertion)
{
    namespace mm = modmesh;
    mm::RadixTree<mm::TimedEntry> radix_tree;
    mm::TimedEntry & entry1 = radix_tree.entry("a");
    entry1.add_time(5.2);
    const mm::RadixTreeNode<mm::TimedEntry> * const node1 = radix_tree.get_current_node();

    mm::TimedEntry & entry2 = radix_tree.entry("b");
    entry2.add_time(10.1);

    EXPECT_EQ(radix_tree.get_unique_node(), 2);

    const mm::RadixTreeNode<mm::TimedEntry> * const node2 = radix_tree.get_current_node();
    EXPECT_EQ(node2->get_prev(), node1);
}

TEST(RadixTree, move_current_pointer)
{
    namespace mm = modmesh;
    mm::RadixTree<mm::TimedEntry> radix_tree;
    mm::TimedEntry & entry1 = radix_tree.entry("a");
    entry1.add_time(5.2);
    const mm::RadixTreeNode<mm::TimedEntry> * const node1 = radix_tree.get_current_node();

    mm::TimedEntry & entry2 = radix_tree.entry("b");
    entry2.add_time(10.1);

    radix_tree.move_current_to_parent();
    const mm::RadixTreeNode<mm::TimedEntry> * const node2 = radix_tree.get_current_node();

    EXPECT_EQ(radix_tree.get_unique_node(), 2);
    EXPECT_EQ(node2->name(), "a");
    EXPECT_EQ(node2, node1);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
