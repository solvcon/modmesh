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

void CallProfilerSerializer::serialize_call_profiler(const CallProfiler & profiler, std::ostream & outstream)
{
    // Example:
    //  {
    //      "radix_tree":
    //      {
    //          "nodes":[
    //              {"key":-1,"name":"","data":{"start_time": 0,"caller_name": "","total_time": 0,"call_count": 0,"is_running": 0},"children":[0]},
    //              {"key":0,"name":"void modmesh::detail::foo1()","data":{"start_time": 17745276708555250,"caller_name": "void modmesh::detail::foo1()","total_time": 61002916,"call_count": 1,"is_running": 1},"children":[1]},
    //              {"key":1,"name":"void modmesh::detail::foo2()","data":{"start_time": 17745276708555458,"caller_name": "void modmesh::detail::foo2()","total_time": 54002250,"call_count": 1,"is_running": 1},"children":[2]},
    //              {"key":2,"name":"void modmesh::detail::foo3()","data":{"start_time": 17745276743555833,"caller_name": "void modmesh::detail::foo3()","total_time": 19001833,"call_count": 1,"is_running": 1},"children":[]}
    //          ],
    //          "current_node":-1,
    //          "unique_id":3
    //      }
    //  }

    outstream << R"({)";
    outstream << R"("radix_tree": )";
    CallProfilerSerializer::serialize_radix_tree(profiler, outstream);
    outstream << R"(})";
}

void CallProfilerSerializer::serialize_radix_tree(const CallProfiler & profiler, std::ostream & outstream)
{
    // Example:
    //  {
    //      "nodes":[
    //          {"key":-1,"name":"","data":{"start_time": 0,"caller_name": "","total_time": 0,"call_count": 0,"is_running": 0},"children":[0]},
    //          {"key":0,"name":"void modmesh::detail::foo1()","data":{"start_time": 17745276708555250,"caller_name": "void modmesh::detail::foo1()","total_time": 61002916,"call_count": 1,"is_running": 1},"children":[1]},
    //          {"key":1,"name":"void modmesh::detail::foo2()","data":{"start_time": 17745276708555458,"caller_name": "void modmesh::detail::foo2()","total_time": 54002250,"call_count": 1,"is_running": 1},"children":[2]},
    //          {"key":2,"name":"void modmesh::detail::foo3()","data":{"start_time": 17745276743555833,"caller_name": "void modmesh::detail::foo3()","total_time": 19001833,"call_count": 1,"is_running": 1},"children":[]}
    //      ],
    //      "current_node":-1,
    //      "unique_id":3
    //  }

    outstream << R"({)";
    outstream << R"("nodes": [)";
    CallProfilerSerializer::serialize_radix_tree_nodes(profiler.radix_tree().get_current_node(), outstream);
    outstream << R"(],)";
    outstream << R"("current_node": )" << profiler.radix_tree().get_current_node()->key() << R"(,)";
    outstream << R"("unique_id": )" << profiler.radix_tree().get_unique_node();
    outstream << R"(})";
}
void CallProfilerSerializer::serialize_radix_tree_nodes(const RadixTreeNode<CallerProfile> * node, std::ostream & outstream)
{
    std::queue<const RadixTreeNode<CallerProfile> *> nodes_buffer;
    nodes_buffer.push(node);
    bool is_first_node = true;

    // Serialize the nodes in a breadth-first manner.
    while (!nodes_buffer.empty())
    {
        int nodes_buffer_size = nodes_buffer.size();
        for (int i = 0; i < nodes_buffer_size; i++)
        {
            const RadixTreeNode<CallerProfile> * current_node = nodes_buffer.front();
            nodes_buffer.pop();
            CallProfilerSerializer::serialize_radix_tree_node(*current_node, is_first_node, outstream);
            is_first_node = false;
            for (const auto & child : current_node->children())
            {
                nodes_buffer.push(child.get());
            }
        }
    }
}

void CallProfilerSerializer::serialize_radix_tree_node(const RadixTreeNode<CallerProfile> & node, bool is_first_node, std::ostream & outstream)
{
    // Example:
    //  {
    //      "key":-1,
    //      "name":"",
    //      "data":{"start_time": 0,"caller_name": "","total_time": 0,"call_count": 0,"is_running": 0},
    //      "children":[0]
    //  }

    // Avoid the trailing comma.
    if (!is_first_node)
    {
        outstream << R"(,)";
    }
    outstream << R"({)";
    outstream << R"("key": )" << node.key() << R"(,)";
    outstream << R"("name": ")" << node.name() << R"(",)";
    outstream << R"("data": )";
    CallProfilerSerializer::serialize_caller_profile(node.data(), outstream);
    outstream << R"(,)";
    outstream << R"("children": [)";
    bool is_first_child = true;
    for (const auto & child : node.children())
    {
        // Avoid the trailing comma.
        if (!is_first_child)
        {
            outstream << R"(,)";
        }
        is_first_child = false;
        outstream << child->key();
    }
    outstream << R"(])";
    outstream << R"(})";
}

void CallProfilerSerializer::serialize_caller_profile(const CallerProfile & profile, std::ostream & outstream)
{
    // Example:
    //  {
    //      "start_time": 0,
    //      "caller_name": "",
    //      "total_time": 0,
    //      "call_count": 0,
    //      "is_running": 0
    //  }

    outstream << R"({)";
    outstream << R"("start_time": )" << profile.start_time.time_since_epoch().count() << R"(,)";
    outstream << R"("caller_name": ")" << profile.caller_name << R"(",)";
    outstream << R"("total_time": )" << profile.total_time.count() << R"(,)";
    outstream << R"("call_count": )" << profile.call_count << R"(,)";
    outstream << R"("is_running": )" << profile.is_running;
    outstream << R"(})";
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
