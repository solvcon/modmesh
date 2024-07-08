/*
 * Copyright (c) 2023, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/toggle/RadixTree.hpp>
#include <modmesh/toggle/profile.hpp>

namespace modmesh
{

void CallProfiler::reset()
{
    while (!m_radix_tree.is_root())
    {
        CallerProfile & profile = m_radix_tree.get_current_node()->data();
        if (profile.is_running)
        {
            profile.stop_stopwatch();
        }
        m_radix_tree.move_current_to_parent();
    }
    m_radix_tree.reset();
    m_cancel_callbacks.clear();
}

// NOLINTNEXTLINE(misc-no-recursion)
void CallProfiler::print_profiling_result(const RadixTreeNode<CallerProfile> & node, const int depth, std::ostream & outstream) const
{
    for (int i = 0; i < depth; ++i)
    {
        outstream << "  ";
    }

    auto profile = node.data();

    if (depth == 0)
    {
        outstream << "Profiling Result" << std::endl;
    }
    else
    {
        outstream << profile.caller_name << " - Total Time: " << profile.total_time.count() / 1e6 << " ms, Call Count: " << profile.call_count << std::endl;
    }

    for (const auto & child : node.children())
    {
        // NOLINTNEXTLINE(misc-no-recursion)
        print_profiling_result(*child, depth + 1, outstream);
    }
}

void CallProfiler::print_statistics(const RadixTreeNode<CallerProfile> & node, std::ostream & outstream)
{
    TimeRegistry::me().clear();

    std::queue<const RadixTreeNode<CallerProfile> *> nodes_buffer;
    for (const auto & child : node.children())
    {
        nodes_buffer.push(child.get());
    }

    // BFS algorithm and put the node information into TimeRegistry.
    while (!nodes_buffer.empty())
    {
        const int nodes_buffer_size = nodes_buffer.size();
        for (int i = 0; i < nodes_buffer_size; ++i)
        {
            const RadixTreeNode<CallerProfile> * current_node = nodes_buffer.front();
            nodes_buffer.pop();
            TimeRegistry::me().add(
                current_node->data().caller_name,
                current_node->data().total_time.count() / 1e9,
                current_node->data().total_time.count() / 1e9,
                current_node->data().call_count);

            for (const auto & child : current_node->children())
            {
                nodes_buffer.push(child.get());
                TimeRegistry::me().add(
                    current_node->data().caller_name,
                    0,
                    -child->data().total_time.count() / 1e9,
                    0);
            }
        }
    }

    // Print the statistics.
    outstream << TimeRegistry::me().detailed_report();

    // Reset the TimeRegistry.
    TimeRegistry::me().clear();
}

void CallProfilerSerializer::serialize_call_profiler(const CallProfiler & profiler, std::ostream & outstream)
{
    // Serialize the RadixTree in the CallProfiler.
    outstream << R"({)" << '\n';
    CallProfilerSerializer::serialize_radix_tree(profiler, outstream);
    outstream << R"(})" << '\n';
}

void CallProfilerSerializer::serialize_radix_tree(const CallProfiler & profiler, std::ostream & outstream)
{
    // Serialize the RadixTree.
    outstream << R"(    "radix_tree": {)" << '\n';
    outstream << R"(        "current_node": )" << profiler.radix_tree().get_current_node()->key() << R"(,)" << '\n';
    outstream << R"(        "unique_id": )" << profiler.radix_tree().get_unique_node() << R"(,)" << '\n';
    CallProfilerSerializer::serialize_id_map(profiler.radix_tree().get_id_map(RadixTree<CallerProfile>::CallProfilerPK()), outstream);
    CallProfilerSerializer::serialize_radix_tree_nodes(profiler.radix_tree().get_current_node(), outstream);
    outstream << R"(    })" << '\n';
}

void CallProfilerSerializer::serialize_id_map(const std::unordered_map<std::string, CallProfilerSerializer::key_type> & id_map, std::ostream & outstream)
{
    // Serialize the unordered_map in RadixTree.
    outstream << R"(        "id_map": {)";

    // If the id_map is empty, close the map at the same line.
    if (id_map.empty())
    {
        outstream << R"(},)" << '\n';
    }
    else
    {
        // Newline after the opening brace.
        outstream << '\n';

        bool is_first = true;
        for (const auto & [key, value] : id_map)
        {
            // Avoid the trailing comma for the first element.
            if (!is_first)
            {
                outstream << R"(,)" << '\n';
            }
            is_first = false;
            outstream << R"(            ")" << key << R"(": )" << value;
        }

        // Newline after the last element.
        outstream << '\n';
        outstream << R"(        },)" << '\n';
    }
}

void CallProfilerSerializer::serialize_radix_tree_nodes(const RadixTreeNode<CallerProfile> * node, std::ostream & outstream)
{
    // Serialize all the RadixTreeNodes in RadixTree in a breadth-first manner.
    outstream << R"(        "nodes": [)";

    // Give each node a unique number
    int unique_node_number = -1;

    std::queue<const RadixTreeNode<CallerProfile> *> nodes_buffer;
    CallProfilerSerializer::node_to_number_map_type node_to_unique_number;

    nodes_buffer.push(node);
    node_to_unique_number[node] = unique_node_number;
    bool is_first_node = true;

    // BFS algorithm
    while (!nodes_buffer.empty())
    {
        const int nodes_buffer_size = nodes_buffer.size();
        for (int i = 0; i < nodes_buffer_size; ++i)
        {
            const RadixTreeNode<CallerProfile> * current_node = nodes_buffer.front();
            nodes_buffer.pop();

            for (const auto & child : current_node->children())
            {
                ++unique_node_number;
                nodes_buffer.push(child.get());
                node_to_unique_number[child.get()] = unique_node_number;
            }

            CallProfilerSerializer::serialize_radix_tree_node(*current_node, is_first_node, node_to_unique_number, outstream);
            is_first_node = false;

            // Remove the node from the map
            node_to_unique_number.erase(current_node);
        }
    }

    // Newline after the last element.
    outstream << '\n';
    outstream << R"(        ])" << '\n';
}

void CallProfilerSerializer::serialize_radix_tree_node(const RadixTreeNode<CallerProfile> & node, bool is_first_node, CallProfilerSerializer::node_to_number_map_type & node_to_unique_number, std::ostream & outstream)
{
    // Serialize the RadixTreeNode to the json format.

    // Avoid the trailing comma for the first node.
    if (!is_first_node)
    {
        outstream << R"(,)" << '\n';
        outstream << R"(            {)" << '\n';
    }
    else
    {
        outstream << R"({)" << '\n';
    }

    outstream << R"(                "unique_number": )" << node_to_unique_number[&node] << R"(,)" << '\n';
    outstream << R"(                "key": )" << node.key() << R"(,)" << '\n';
    outstream << R"(                "name": ")" << node.name() << R"(",)" << '\n';
    CallProfilerSerializer::serialize_caller_profile(node.data(), outstream);
    CallProfilerSerializer::serialize_radix_tree_node_children(node.children(), node_to_unique_number, outstream);

    outstream << R"(            })";
}

void CallProfilerSerializer::serialize_radix_tree_node_children(const CallProfilerSerializer::child_list_type & children, CallProfilerSerializer::node_to_number_map_type & node_to_unique_number, std::ostream & outstream)
{
    // Serialize the children list in RadixTreeNode.
    outstream << R"(                children": [)";

    // If the children list is empty, close the list at the same line.
    if (children.empty())
    {
        outstream << R"(])" << '\n';
    }

    else
    {
        outstream << '\n';

        bool is_first_child = true;
        for (const auto & child : children)
        {
            // Avoid the trailing comma.
            if (!is_first_child)
            {
                outstream << R"(,)" << '\n';
            }
            is_first_child = false;
            outstream << R"(                    )" << node_to_unique_number[child.get()] << R"()";
        }
        outstream << '\n';
        outstream << R"(                ])" << '\n';
    }
}

void CallProfilerSerializer::serialize_caller_profile(const CallerProfile & profile, std::ostream & outstream)
{
    // Serialize the CallerProfile to the json format.
    outstream << R"(                "data": {)" << '\n';
    outstream << R"(                    "start_time": )" << profile.start_time.time_since_epoch().count() << R"(,)" << '\n';
    outstream << R"(                    "caller_name": ")" << profile.caller_name << R"(",)" << '\n';
    outstream << R"(                    "total_time": )" << profile.total_time.count() << R"(,)" << '\n';
    outstream << R"(                    "call_count": )" << profile.call_count << R"(,)" << '\n';
    outstream << R"(                    "is_running": )" << profile.is_running << '\n';
    outstream << R"(                },)" << '\n';
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
