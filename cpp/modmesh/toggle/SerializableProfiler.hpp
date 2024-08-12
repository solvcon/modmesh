#pragma once

/*
 * Copyright (c) 2024, Chun-Shih Chang <austin20463@gmail.com>
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
#include <modmesh/serialization/SerializableItem.hpp>

namespace modmesh
{

class SerializableRadixTreeNode : SerializableItem
{

public:

    using child_list_type = std::vector<SerializableRadixTreeNode>;
    using key_type = typename RadixTree<CallerProfile>::key_type;

    SerializableRadixTreeNode() = default;
    SerializableRadixTreeNode(SerializableRadixTreeNode const &) = default;
    SerializableRadixTreeNode(SerializableRadixTreeNode &&) = default;
    SerializableRadixTreeNode & operator=(SerializableRadixTreeNode const &) = default;
    SerializableRadixTreeNode & operator=(SerializableRadixTreeNode &&) = default;
    ~SerializableRadixTreeNode() = default;
    SerializableRadixTreeNode(const RadixTreeNode<CallerProfile> * node)
        : m_key(node->key())
        , m_name(node->name())
        , m_total_time(node->data().stable_total_time)
        , m_call_count(node->data().stable_call_count)
    {
        for (const auto & child : node->children())
        {
            if (child->data().stable_call_count > 0)
            {
                m_children.push_back(SerializableRadixTreeNode(child.get()));
            }
        }
    }

    MM_DECL_SERIALIZABLE(
        register_member("key", m_key);
        register_member("name", m_name);
        register_member("total_time", static_cast<double>(m_total_time.count()));
        register_member("call_count", m_call_count);
        register_member("children", m_children);)

private:

    /*
     * In T data, 'bool is_running' and
     * 'std::chrono::high_resolution_clock::time_point start_time' are not serialized
     * because they are useless when deserialize the object.
     */
    key_type m_key;
    std::string m_name;
    std::chrono::nanoseconds m_total_time;
    int m_call_count;
    child_list_type m_children;
}; /* end class SerializableRadixTreeNode */

class SerializableRadixTree : SerializableItem
{

public:

    using key_type = typename RadixTree<CallerProfile>::key_type;
    SerializableRadixTree() = default;
    SerializableRadixTree(SerializableRadixTree const &) = default;
    SerializableRadixTree(SerializableRadixTree &&) = default;
    SerializableRadixTree & operator=(SerializableRadixTree const &) = default;
    SerializableRadixTree & operator=(SerializableRadixTree &&) = default;
    ~SerializableRadixTree() = default;
    SerializableRadixTree(const RadixTree<CallerProfile> & radix_tree)
        : m_root(radix_tree.get_root())
        , m_id_map(radix_tree.get_stable_id_map())
        , m_unique_id(radix_tree.get_stable_unique_node())
    {
    }

    MM_DECL_SERIALIZABLE(
        register_member("radix_tree", m_root);
        register_member("id_map", m_id_map);
        register_member("unique_id", m_unique_id);)

private:

    SerializableRadixTreeNode m_root;
    std::unordered_map<std::string, key_type> m_id_map;
    key_type m_unique_id;
}; /* end class SerializableRadixTree */

/// Utility to serialize and deserialize CallProfiler.
class CallProfilerSerializer
{

public:

    // It returns the json format of the CallProfiler.
    static std::string serialize(const CallProfiler & profiler)
    {
        auto radix_tree_curent_node = profiler.radix_tree().get_current_node();
        auto radix_tree_root = profiler.radix_tree().get_root();
        SerializableRadixTree serializable_radix_tree(profiler.radix_tree());
        return serializable_radix_tree.to_json();
    }

}; /* end class CallProfilerSerializer */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
