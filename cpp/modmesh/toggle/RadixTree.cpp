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

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
