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

TEST(RadixTree, insertion)
{
    using namespace modmesh;
    RadixTree<TimedEntry> radixTree;
    TimedEntry& entry1 = radixTree.entry("a");
    entry1.add_time(5.2);

    EXPECT_EQ(entry1.count(), 1);
    EXPECT_DOUBLE_EQ(entry1.time(), 5.2);
    EXPECT_EQ(radixTree.getUniqueNode(), 1);

    const RadixTreeNode<TimedEntry>* const node = radixTree.getCurrentNode();
    EXPECT_EQ(node->getName(), "a");
    EXPECT_EQ(node->Info(), "a - Count: 1 - Time: 5.2");
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
