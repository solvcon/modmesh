#include <modmesh/toggle/RadixTree.hpp>
#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(RadixTree, construction)
{
    using namespace modmesh;
    RadixTree<TimedEntry> radixTree;
    EXPECT_EQ(radixTree.getUniqueNode(), 0);
}

TEST(RadixTree, single_insertion)
{
    using namespace modmesh;
    RadixTree<TimedEntry> radixTree;
    TimedEntry & entry1 = radixTree.entry("a");
    entry1.add_time(5.2);

    EXPECT_EQ(entry1.count(), 1);
    EXPECT_DOUBLE_EQ(entry1.time(), 5.2);
    EXPECT_EQ(radixTree.getUniqueNode(), 1);

    const RadixTreeNode<TimedEntry> * const node = radixTree.getCurrentNode();
    EXPECT_EQ(node->getName(), "a");
    EXPECT_EQ(node->Info(), "a - Count: 1 - Time: 5.2");
}

TEST(RadixTree, multiple_insertion)
{
    using namespace modmesh;
    RadixTree<TimedEntry> radixTree;
    TimedEntry & entry1 = radixTree.entry("a");
    entry1.add_time(5.2);
    const RadixTreeNode<TimedEntry> * const node1 = radixTree.getCurrentNode();

    TimedEntry & entry2 = radixTree.entry("b");
    entry2.add_time(10.1);

    EXPECT_EQ(radixTree.getUniqueNode(), 2);

    const RadixTreeNode<TimedEntry> * const node2 = radixTree.getCurrentNode();
    EXPECT_EQ(node2->getPrev(), node1);
}

TEST(RadixTree, move_current_pointer)
{
    using namespace modmesh;
    RadixTree<TimedEntry> radixTree;
    TimedEntry & entry1 = radixTree.entry("a");
    entry1.add_time(5.2);
    const RadixTreeNode<TimedEntry> * const node1 = radixTree.getCurrentNode();

    TimedEntry & entry2 = radixTree.entry("b");
    entry2.add_time(10.1);

    radixTree.moveCurrentToParent();
    const RadixTreeNode<TimedEntry> * const node2 = radixTree.getCurrentNode();

    EXPECT_EQ(radixTree.getUniqueNode(), 2);
    EXPECT_EQ(node2->getName(), "a");
    EXPECT_EQ(node2, node1);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
