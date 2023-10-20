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

namespace modmesh
{
/**
 * Simple Timed Entry
 */
class TimedEntry {
public:
    size_t count() const { return m_count; }
    double time() const { return m_time; }

    TimedEntry& add_time(double time) {
        ++m_count;
        m_time += time;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const TimedEntry& entry) {
        os << "Count: " << entry.count() << " - Time: " << entry.time();
        return os;
    }

private:
    size_t m_count = 0;
    double m_time = 0.0;
}; /* end class TimedEntry */

template <typename T>
class RadixTreeNode {
private:
    using ChildList = std::list<std::unique_ptr<RadixTreeNode>>;

public:
    RadixTreeNode() : prev(nullptr) {};
    RadixTreeNode(std::string name, int key) : m_name(std::move(name)), m_key(key), prev(nullptr) {}
    
    const int& getKey() const { return m_key; }
    const std::string& getName() const { return m_name; }
    T& getData() { return m_data; }
    const T& getData() const { return m_data; }
    const ChildList& getChildren() const { return m_children; }

    // Add a child node
    RadixTreeNode<T>* addChild(std::string childName, int childKey) {
        auto newChild = std::make_unique<RadixTreeNode<T>>(std::move(childName), std::move(childKey));
        newChild->prev = this;
        m_children.push_back(std::move(newChild));
        return m_children.back().get();
    }

    // Get a child node with a given key
    RadixTreeNode<T>* getChild(int key) const {
        auto it = std::find_if(m_children.begin(), m_children.end(), [&](const auto& child) {
            return child->getKey() == key;
        });
        return (it != m_children.end()) ? it->get() : nullptr;
    }
    
    // Get prev node in the tree
    RadixTreeNode<T>* getPrev() const {
        return prev;
    }

    // // print node information
    std::string Info() const {
        std::ostringstream oss;
        if (nullptr != prev) {
            oss << getName() << " - ";
            oss << getData();
        }
        return oss.str();
    }

private:
    int m_key = -1;
    std::string m_name;
    T m_data;
    ChildList m_children;
    RadixTreeNode<T>* prev;
}; /* end class RadixTreeNode */

/*
Ref:
https://kalkicode.com/radix-tree-implementation
https://www.algotree.org/algorithms/trie/
*/
template <typename T>
class RadixTree {
public:
    RadixTree() : m_root(std::make_unique<RadixTreeNode<T>>()), m_currentNode(m_root.get()) {}

    T& entry(const std::string& name) {
        int id = getId(name);

        RadixTreeNode<T>* child = m_currentNode->getChild(id);

        if (!child) {
            m_currentNode = m_currentNode->addChild(name, id);
        } else {
            m_currentNode = child;
        }

        return m_currentNode->getData();
    }

    void moveCurrentToParent() {
        if (m_currentNode != m_root.get()) {
            m_currentNode = m_currentNode->getPrev();
        }
    }

    const RadixTreeNode<T>* getCurrentNode() const { return m_currentNode; }
    const int getUniqueNode() const {
        return m_unique_id;
    }

    void print() const {
        printRecursive(m_root.get(), 0);
    }

private:
    void printRecursive(const RadixTreeNode<T>* node, int depth) const {
        for (int i = 0; i < depth; ++i) {
            std::cout << "  ";
        }

        std::cout << node->Info();

        for (const auto& child : node->getChildren()) {
            printRecursive(child.get(), depth + 1);
        }
    }

    int getId(const std::string& name) {
        auto [it, inserted] = m_id_map.try_emplace(name, m_unique_id++);
        return it->second;
    }

    std::unique_ptr<RadixTreeNode<T>> m_root;
    RadixTreeNode<T>* m_currentNode;
    std::unordered_map<std::string, int> m_id_map;
    int m_unique_id = 0;
}; /* end class RadixTree */

} /* end namespace modmesh */
