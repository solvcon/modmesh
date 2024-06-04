#include <gtest/gtest.h>
#include <thread>
#include <algorithm>
#include <unordered_set>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#define CALLPROFILER 1
#include <modmesh/toggle/RadixTree.hpp>
namespace modmesh
{

namespace detail
{
class CallProfilerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CallProfiler & profiler = CallProfiler::instance();
        pProfiler = &profiler;
    }

    RadixTree<CallerProfile> & radix_tree()
    {
        return pProfiler->m_radix_tree;
    }

    CallProfiler * pProfiler;

public:
    using node_to_number_map_type = std::unordered_map<const RadixTreeNode<CallerProfile> *, int>;
    using number_to_node_map_type = std::unordered_map<int, const RadixTreeNode<CallerProfile> *>;
    void check_id_map_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines, bool functionIsExist);
    void check_radix_tree_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines);
    void check_call_profiler_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines);
    void check_radix_tree_nodes_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines);
    std::vector<std::vector<std::string>> split_nodes_info(const std::vector<std::string> & lines);
    void BFS_radix_tree(const RadixTreeNode<CallerProfile> * node, node_to_number_map_type & node_to_unique_number, number_to_node_map_type & unique_number_to_node);
};

void CallProfilerTest::check_call_profiler_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines)
{
    int numOfLines = lines.size();

    // Expect the first and last line are parantheses.
    auto lineCallProfilerBegin = "{";
    auto lineCallProfilerEnd = "}";
    EXPECT_EQ(lines[0], lineCallProfilerBegin);
    EXPECT_EQ(lines[numOfLines - 1], lineCallProfilerEnd);

    // Check the radix_tree serialization.
    check_radix_tree_serialization(profiler, lines);
}

void CallProfilerTest::check_radix_tree_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines)
{
    int numOfLines = lines.size();

    // Expect the second and second last line are the begin and the end of radix_tree.
    auto lineRadixTreeBegin = R"(    "radix_tree": {)";
    auto lineRadixTreeEnd = R"(    })";
    EXPECT_EQ(lines[1], lineRadixTreeBegin);
    EXPECT_EQ(lines[numOfLines - 2], lineRadixTreeEnd);

    // Expect the third line is the current_node.
    auto lineCurrentNode = R"(        "current_node": )";
    auto CurrentNodeValue = std::to_string(profiler.radix_tree().get_current_node()->key());
    EXPECT_EQ(lines[2], lineCurrentNode + CurrentNodeValue + ",");

    // Expect the fourth line is the unique_id.
    auto lineUniqueId = R"(        "unique_id": )";
    auto UniqueIdValue = std::to_string(profiler.radix_tree().get_unique_node());
    EXPECT_EQ(lines[3], lineUniqueId + UniqueIdValue + ",");

    bool functionIsExist = profiler.radix_tree().get_unique_node() > 0;

    // Check the id_map serialization.
    check_id_map_serialization(profiler, lines, functionIsExist);

    // Check the nodes serialization.
    check_radix_tree_nodes_serialization(profiler, lines);
}

void CallProfilerTest::check_id_map_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines, bool functionIsExist)
{
    int numOfLines = lines.size();

    // Expect the id_map contains no function.
    if (!functionIsExist)
    {
        // Expect the fifth line is the id_map.
        auto lineIdMap = R"(        "id_map": {},)";
        EXPECT_EQ(lines[4], lineIdMap);

        // Expect the line after the end of the id_map is the begin of nodes.
        auto lineNodesBegin = R"(        "nodes": [{)";
        EXPECT_EQ(lines[5], lineNodesBegin);
    }

    else
    {
        // Expect the fifth line is the begin of id_map.
        auto lineIdMapBegin = R"(        "id_map": {)";
        EXPECT_EQ(lines[4], lineIdMapBegin);

        auto lineIdMapEnd = R"(        },)";
        int lineIdMapEndIndex = std::find(lines.begin(), lines.end(), lineIdMapEnd) - lines.begin();

        // Expect the end of the id_map is exist.
        EXPECT_NE(lineIdMapEndIndex, numOfLines);

        auto lineNodesBegin = R"(        "nodes": [{)";
        // Expect the line after the end of the id_map is the begin of nodes.
        EXPECT_EQ(lines[lineIdMapEndIndex + 1], lineNodesBegin);

        // Store the serialization of the key-value pair in the hash set.
        std::unordered_set<std::string> idMapPairStrings;
        for (int i = 5; i < lineIdMapEndIndex; i++)
        {
            if (i < lineIdMapEndIndex - 1)
            {
                // Expect the trailing comma for the last element.
                EXPECT_EQ(lines[i].back(), ',');
                idMapPairStrings.insert(lines[i]);
            }
            else
            {
                // Expect the last element has no trailing comma.
                EXPECT_NE(lines[i].back(), ',');
                idMapPairStrings.insert(lines[i] + ",");
            }
        }

        auto id_map = profiler.radix_tree().get_id_map(RadixTree<CallerProfile>::CallProfilerPK());

        for (auto [key, value] : id_map)
        {
            // Expect the key-value pair is in the hash set.
            auto line = R"(            ")" + key + R"(": )" + std::to_string(value) + ",";
            EXPECT_TRUE(idMapPairStrings.find(line) != idMapPairStrings.end());
            idMapPairStrings.erase(line);
        }
    }
}

void CallProfilerTest::check_radix_tree_nodes_serialization(const CallProfiler & profiler, const std::vector<std::string> & lines)
{
    int numOfLines = lines.size();

    // Expect there exists the begin of nodes.
    auto lineNodesBegin = R"(        "nodes": [{)";
    int lineNodesBeginIndex = std::find(lines.begin(), lines.end(), lineNodesBegin) - lines.begin();
    EXPECT_NE(lineNodesBeginIndex, numOfLines);

    // Expect the third last line is the end of nodes.
    auto lineNodesEnd = R"(        ])";
    EXPECT_EQ(lines[numOfLines - 3], lineNodesEnd);

    // Split the nodes serialization into a vector of a vector of string.
    std::vector<std::vector<std::string>> nodesInfo = split_nodes_info(lines);

    // BFS and number the radix tree nodes
    node_to_number_map_type node_to_unique_number;
    number_to_node_map_type unique_number_to_node;
    BFS_radix_tree(profiler.radix_tree().get_current_node(), node_to_unique_number, unique_number_to_node);

    // Check the serialization of each node.
    int numOfNodes = nodesInfo.size();
    for (int i = 0; i < numOfNodes; ++i)
    {
        // The serialization of the i-th node.
        auto linesNodeInfo = nodesInfo[i];
        int numOfLinesNodeInfo = linesNodeInfo.size();

        // The pointer to the i-th node.
        auto node = unique_number_to_node[i - 1];

        // Expect the first line is the begin of the node.
        auto lineNodeBegin = R"(            {)";
        EXPECT_EQ(linesNodeInfo[0], lineNodeBegin);

        // Expect the last line is the end of the node.
        auto lineNodeEnd = R"(            },)";
        EXPECT_EQ(linesNodeInfo[numOfLinesNodeInfo - 1], lineNodeEnd);

        // Expect the second line is the unique_number.
        auto lineUniqueNumber = R"(                "unique_number": )";
        auto UniqueNumberValue = std::to_string(i - 1);
        EXPECT_EQ(linesNodeInfo[1], lineUniqueNumber + UniqueNumberValue + ",");

        // Expect the third line is the key.
        auto lineKey = R"(                "key": )";
        auto KeyValue = std::to_string(node->key());
        EXPECT_EQ(linesNodeInfo[2], lineKey + KeyValue + ",");

        // Expect the fourth line is the name.
        auto lineName = R"(                "name": ")";
        auto NameValue = node->name();
        EXPECT_EQ(linesNodeInfo[3], lineName + NameValue + R"(",)");

        // Expect the line 5 ~ 11 are the data.
        auto lineDataBegin = R"(                "data": {)";
        EXPECT_EQ(linesNodeInfo[4], lineDataBegin);

        auto lineStartTime = R"(                    "start_time": )";
        auto StartTimeValue = std::to_string(node->data().start_time.time_since_epoch().count());
        EXPECT_EQ(linesNodeInfo[5], lineStartTime + StartTimeValue + ",");

        auto lineCallerName = R"(                    "caller_name": ")";
        auto CallerNameValue = node->data().caller_name;
        EXPECT_EQ(linesNodeInfo[6], lineCallerName + CallerNameValue + R"(",)");

        auto lineTotalTime = R"(                    "total_time": )";
        auto TotalTimeValue = std::to_string(node->data().total_time.count());
        EXPECT_EQ(linesNodeInfo[7], lineTotalTime + TotalTimeValue + ",");

        auto lineCallCount = R"(                    "call_count": )";
        auto CallCountValue = std::to_string(node->data().call_count);
        EXPECT_EQ(linesNodeInfo[8], lineCallCount + CallCountValue + ",");

        auto lineIsRunning = R"(                    "is_running": )";
        auto IsRunningValue = std::to_string(node->data().is_running);
        EXPECT_EQ(linesNodeInfo[9], lineIsRunning + IsRunningValue);

        auto lineDataEnd = R"(                },)";
        EXPECT_EQ(linesNodeInfo[10], lineDataEnd);

        bool childrenIsEmpty = node->children().empty();
        // If the children list is empty, expect the children list is closed at the same line (line 12).
        if (childrenIsEmpty)
        {
            auto lineChildren = R"(                children": [])";
            EXPECT_EQ(linesNodeInfo[11], lineChildren);
        }
        else
        {
            // line 12 is the begin of children.
            auto lineChildrenBegin = R"(                children": [)";
            EXPECT_EQ(linesNodeInfo[11], lineChildrenBegin);

            // The second last line is the end of children.
            auto lineChildrenEnd = R"(                ])";
            EXPECT_EQ(linesNodeInfo[numOfLinesNodeInfo - 2], lineChildrenEnd);

            // Check the trailing comma for the children list.
            for (int j = 12; j < numOfLinesNodeInfo - 3; j++)
            {
                EXPECT_EQ(linesNodeInfo[j].back(), ',');
            }
            // Expect the last element has no trailing comma.
            EXPECT_NE(linesNodeInfo[numOfLinesNodeInfo - 3].back(), ',');

            // Check the unique number of children list is correct.
            std::unordered_set<int> childrenUniqueNumbers;
            for (int j = 12; j < numOfLinesNodeInfo - 2; j++)
            {
                std::string uniqueNumberString = linesNodeInfo[j];
                if (uniqueNumberString.back() == ',')
                    uniqueNumberString.pop_back();
                int uniqueNumber = std::stoi(uniqueNumberString);
                childrenUniqueNumbers.insert(uniqueNumber);
            }

            for (const auto & child : node->children())
            {
                int uniqueNumber = node_to_unique_number[child.get()];
                EXPECT_TRUE(childrenUniqueNumbers.find(uniqueNumber) != childrenUniqueNumbers.end());
                childrenUniqueNumbers.erase(uniqueNumber);
            }
        }
    }
}

std::vector<std::vector<std::string>> CallProfilerTest::split_nodes_info(const std::vector<std::string> & lines)
{
    int numOfLines = lines.size();
    // Split the nodes serialization into a vector of strings.
    std::vector<std::vector<std::string>> nodesInfo;
    auto lineNodesBegin = R"(        "nodes": [{)";
    int curNodeInfoBegin = std::find(lines.begin(), lines.end(), lineNodesBegin) - lines.begin();
    int curNodeInfoEnd = 0;

    while (curNodeInfoEnd < numOfLines - 4)
    {
        std::vector<std::string> nodeInfo;
        curNodeInfoEnd = std::find(lines.begin() + curNodeInfoBegin, lines.end(), R"(            },)") - lines.begin();
        if (curNodeInfoEnd == numOfLines)
        {
            curNodeInfoEnd = numOfLines - 4;
            // Expect the last node has no trailing comma.
            EXPECT_NE(lines[curNodeInfoEnd].back(), ',');
        }

        // Unify the first line and last line of the node to make the check easier.
        nodeInfo.push_back(R"(            {)");
        for (int i = curNodeInfoBegin + 1; i <= curNodeInfoEnd; i++)
        {
            if (i == numOfLines - 4)
                nodeInfo.push_back(lines[i] + ",");
            else
                nodeInfo.push_back(lines[i]);
        }
        nodesInfo.push_back(nodeInfo);
        curNodeInfoBegin = curNodeInfoEnd + 1;
    }
    return nodesInfo;
}

void CallProfilerTest::BFS_radix_tree(const RadixTreeNode<CallerProfile> * node, node_to_number_map_type & node_to_unique_number, number_to_node_map_type & unique_number_to_node)
{
    // BFS the radix tree and number the nodes.
    std::queue<const RadixTreeNode<CallerProfile> *> nodes_buffer;
    nodes_buffer.push(node);
    node_to_unique_number[node] = -1;
    unique_number_to_node[-1] = node;
    int unique_node_number = -1;

    while (!nodes_buffer.empty())
    {
        const int nodes_buffer_size = nodes_buffer.size();
        for (int i = 0; i < nodes_buffer_size; ++i)
        {
            const RadixTreeNode<CallerProfile> * cur_node = nodes_buffer.front();
            nodes_buffer.pop();
            for (const auto & child : cur_node->children())
            {
                nodes_buffer.push(child.get());
                // Store the key and value in the two hash maps.
                node_to_unique_number[child.get()] = ++unique_node_number;
                unique_number_to_node[unique_node_number] = child.get();
            }
        }
    }
}

constexpr int uniqueTime1 = 19;
constexpr int uniqueTime2 = 35;
constexpr int uniqueTime3 = 7;

void func3()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime1)
    {
        // use busy loop to get a precise duration
    }
}

void func2()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime2)
    {
        // use busy loop to get a precise duration
    }
    func3();
}

void func1()
{
    USE_CALLPROFILER_PROFILE_THIS_FUNCTION();
    func2();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime3)
    {
        // use busy loop to get a precise duration
    }
}

std::vector<std::string> split_str(const std::string & s, char delim)
{
    // Convert a string to a vector of strings by splitting it with a delimiter.
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        result.push_back(item);
    }
    return result;
}

TEST_F(CallProfilerTest, test_serialization_1)
{
    /* Here is the expected format of output:
{
    "radix_tree": {
        "current_node": -1,
        "unique_id": 7,
        "id_map": {
            "void modmesh::detail::func3()": 2,
            "void modmesh::detail::func2()": 1,
            "void modmesh::detail::func1()": 0
        },
        "nodes": [{
                "unique_number": -1,
                "key": -1,
                "name": "",
                "data": {
                    "start_time": 0,
                    "caller_name": "",
                    "total_time": 0,
                    "call_count": 0,
                    "is_running": 0
                },
                children": [
                    0,
                    1,
                    2
                ]
            },
            {
                "unique_number": 0,
                "key": 0,
                "name": "void modmesh::detail::func1()",
                "data": {
                    "start_time": 2074704984391166,
                    "caller_name": "void modmesh::detail::func1()",
                    "total_time": 61000834,
                    "call_count": 1,
                    "is_running": 1
                },
                children": [
                    3
                ]
            },
            {
                "unique_number": 1,
                "key": 1,
                "name": "void modmesh::detail::func2()",
                "data": {
                    "start_time": 2074705045392250,
                    "caller_name": "void modmesh::detail::func2()",
                    "total_time": 54001000,
                    "call_count": 1,
                    "is_running": 1
                },
                children": [
                    4
                ]
            },
            {
                "unique_number": 2,
                "key": 2,
                "name": "void modmesh::detail::func3()",
                "data": {
                    "start_time": 2074705118393708,
                    "caller_name": "void modmesh::detail::func3()",
                    "total_time": 38000208,
                    "call_count": 2,
                    "is_running": 1
                },
                children": []
            },
            {
                "unique_number": 3,
                "key": 1,
                "name": "void modmesh::detail::func2()",
                "data": {
                    "start_time": 2074704984391458,
                    "caller_name": "void modmesh::detail::func2()",
                    "total_time": 54000417,
                    "call_count": 1,
                    "is_running": 1
                },
                children": [
                    5
                ]
            },
            {
                "unique_number": 4,
                "key": 2,
                "name": "void modmesh::detail::func3()",
                "data": {
                    "start_time": 2074705080392791,
                    "caller_name": "void modmesh::detail::func3()",
                    "total_time": 19000417,
                    "call_count": 1,
                    "is_running": 1
                },
                children": []
            },
            {
                "unique_number": 5,
                "key": 2,
                "name": "void modmesh::detail::func3()",
                "data": {
                    "start_time": 2074705019391625,
                    "caller_name": "void modmesh::detail::func3()",
                    "total_time": 19000125,
                    "call_count": 1,
                    "is_running": 1
                },
                children": []
            }
        ]
    }
}
    */
    pProfiler->reset();

    func1();
    func2();
    func3();
    func3();

    std::stringstream ss;
    CallProfilerSerializer::serialize(*pProfiler, ss);
    std::vector<std::string> lines = split_str(ss.str(), '\n');
    //std::cout << ss.str() << std::endl;
    check_call_profiler_serialization(*pProfiler, lines);
}

TEST_F(CallProfilerTest, test_serialization_2)
{
    /* Here is the expected format of output:
{
    "radix_tree": {
        "current_node": -1,
        "unique_id": 0,
        "id_map": {},
        "nodes": [{
                "unique_number": -1,
                "key": -1,
                "name": "",
                "data": {
                    "start_time": 0,
                    "caller_name": "",
                    "total_time": 0,
                    "call_count": 0,
                    "is_running": 0
                },
                children": []
            }
        ]
    }
}
    */
    pProfiler->reset();

    std::stringstream ss;
    CallProfilerSerializer::serialize(*pProfiler, ss);
    std::vector<std::string> lines = split_str(ss.str(), '\n');
    //std::cout << ss.str() << std::endl;
    check_call_profiler_serialization(*pProfiler, lines);
}

} // namespace detail
} // namespace modmesh