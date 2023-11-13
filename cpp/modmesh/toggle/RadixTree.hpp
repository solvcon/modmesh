#pragma once

/*
 * Copyright (c) 2023, Quentin Tsai <quentin.tsai.tw@gmail.com>
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

#include <sstream>
#include <iostream>
#include <memory>
#include <list>
#include <unordered_map>
#include <stack>
#include <algorithm>

namespace modmesh
{
/**
 * Simple Timed Entry
 */
class TimedEntry
{
public:
    size_t count() const { return m_count; }
    double time() const { return m_time; }

    TimedEntry & add_time(double time)
    {
        ++m_count;
        m_time += time;
        return *this;
    }

    friend std::ostream & operator<<(std::ostream & os, const TimedEntry & entry)
    {
        os << "Count: " << entry.count() << " - Time: " << entry.time();
        return os;
    }

private:
    size_t m_count = 0;
    double m_time = 0.0;
}; /* end class TimedEntry */

template <typename T>
class RadixTreeNode
{
public:

    using child_list_type = std::list<std::unique_ptr<RadixTreeNode<T>>>;
    using key_type = int32_t;

    static_assert(std::is_integral_v<key_type> && std::is_signed_v<key_type>, "signed integral required");

    RadixTreeNode(std::string const & name, key_type key)
        : m_name(name)
        , m_key(key)
        , m_prev(nullptr)
    {
    }

    RadixTreeNode() = default;
    RadixTreeNode(RadixTreeNode const &) = default;
    RadixTreeNode(RadixTreeNode &&) = default;
    RadixTreeNode & operator=(RadixTreeNode const &) = default;
    RadixTreeNode & operator=(RadixTreeNode &&) = default;
    ~RadixTreeNode() = default;

    key_type key() const { return m_key; }
    const std::string & name() const { return m_name; }
    T & data() { return m_data; }
    const T & data() const { return m_data; }
    const child_list_type & children() const { return m_children; }

    RadixTreeNode<T> * add_child(std::string const & name, key_type key)
    {
        auto new_child = std::make_unique<RadixTreeNode<T>>(name, key);
        new_child->m_prev = this;
        m_children.push_back(std::move(new_child));
        return m_children.back().get();
    }

    RadixTreeNode<T> * get_child(key_type key) const
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [&](const auto & child)
                               { return child->key() == key; });
        return (it != m_children.end()) ? it->get() : nullptr;
    }

    RadixTreeNode<T> * get_prev() const
    {
        return m_prev;
    }

private:
    key_type m_key = -1;
    std::string m_name;
    T m_data;
    child_list_type m_children;
    RadixTreeNode<T> * m_prev = nullptr;
}; /* end class RadixTreeNode */

/*
Ref:
https://kalkicode.com/radix-tree-implementation
https://www.algotree.org/algorithms/trie/
*/
template <typename T>
class RadixTree
{
public:
    using key_type = typename RadixTreeNode<T>::key_type;
    RadixTree()
        : m_root(std::make_unique<RadixTreeNode<T>>())
        , m_current_node(m_root.get())
    {
    }

    T & entry(const std::string & name)
    {
        key_type id = get_id(name);

        RadixTreeNode<T> * child = m_current_node->get_child(id);

        if (!child)
        {
            m_current_node = m_current_node->add_child(name, id);
        }
        else
        {
            m_current_node = child;
        }

        return m_current_node->data();
    }

    void move_current_to_parent()
    {
        if (m_current_node != m_root.get())
        {
            m_current_node = m_current_node->get_prev();
        }
    }

    RadixTreeNode<T> * get_current_node() const { return m_current_node; }
    key_type get_unique_node() const { return m_unique_id; }

private:
    key_type get_id(const std::string & name)
    {
        auto [it, inserted] = m_id_map.try_emplace(name, m_unique_id++);
        return it->second;
    }

    // FIXME: Use raw pointer for m_root.
    std::unique_ptr<RadixTreeNode<T>> m_root;
    RadixTreeNode<T> * m_current_node;
    std::unordered_map<std::string, key_type> m_id_map;
    key_type m_unique_id = 0;
}; /* end class RadixTree */

} /* end namespace modmesh */
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
