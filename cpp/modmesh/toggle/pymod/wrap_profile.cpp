/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/toggle/pymod/toggle_pymod.hpp> // Must be the first include.
#include <modmesh/toggle/SerializableProfiler.hpp>
#include <modmesh/modmesh.hpp>
#include <queue>
#include <unordered_map>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapWrapperProfilerStatus
    : public WrapBase<WrapWrapperProfilerStatus, WrapperProfilerStatus>
{

public:

    friend root_base_type;

protected:

    WrapWrapperProfilerStatus(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            // clang-format off
            .def_property_readonly_static("me", [](py::object const &) -> wrapped_type& { return wrapped_type::me(); })
            // clang-format on
            .def_property_readonly("enabled", &wrapped_type::enabled)
            .def("enable", &wrapped_type::enable)
            .def("disable", &wrapped_type::disable)
            //
            ;

        mod.attr("wrapper_profiler_status") = mod.attr("WrapperProfilerStatus").attr("me");
    }

}; /* end class WrapWrapperTimerStatus */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapStopWatch
    : public WrapBase<WrapStopWatch, StopWatch>
{

public:

    friend root_base_type;

protected:

    WrapStopWatch(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "me",
                [](py::object const &) -> wrapped_type &
                { return wrapped_type::me(); })
            .def("lap", &wrapped_type::lap)
            .def_property_readonly("duration", &wrapped_type::duration)
            .def_property_readonly_static(
                "resolution",
                [](py::object const &)
                { return wrapped_type::resolution(); })
            //
            ;

        mod.attr("stop_watch") = mod.attr("StopWatch").attr("me");
    }

}; /* end class WrapStopWatch */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapTimedEntry
    : public WrapBase<WrapTimedEntry, TimedEntry>
{

public:

    friend root_base_type;

protected:

    WrapTimedEntry(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly("count", &wrapped_type::count)
            .def_property_readonly("time", &wrapped_type::time)
            .def("start", &wrapped_type::start)
            .def("stop", &wrapped_type::stop)
            .def("add_time", &wrapped_type::add_time, py::arg("time"))
            //
            ;
    }

}; /* end class WrapTimedEntry */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapTimeRegistry
    : public WrapBase<WrapTimeRegistry, TimeRegistry>
{

public:

    friend root_base_type;

protected:

    WrapTimeRegistry(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "me",
                [](py::object const &) -> wrapped_type &
                { return wrapped_type::me(); })
            .def("clear", &wrapped_type::clear)
            .def("entry", &wrapped_type::entry, py::arg("name"))
            .def(
                "add_time",
                static_cast<void (wrapped_type::*)(std::string const &, double)>(&wrapped_type::add),
                py::arg("name"),
                py::arg("time"))
            .def_property_readonly("names", &wrapped_type::names)
            .def("report", &wrapped_type::report)
            //
            ;

        mod.attr("time_registry") = mod.attr("TimeRegistry").attr("me");
    }

}; /* end class WrapTimeRegistry */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapCallProfiler : public WrapBase<WrapCallProfiler, CallProfiler>
{
public:
    friend root_base_type;
    static pybind11::dict result(CallProfiler & profiler);

protected:
    WrapCallProfiler(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "instance",
                [](py::object const &) -> wrapped_type &
                { return wrapped_type::instance(); })
            .def(
                "stat",
                [](CallProfiler & profiler)
                {
                    std::stringstream ss;
                    profiler.print_statistics(ss);
                    return ss.str();
                })
            .def("result", &result)
            .def("reset", &wrapped_type::reset)
            .def("serialize", &CallProfilerSerializer::serialize);

        mod.attr("call_profiler") = mod.attr("CallProfiler").attr("instance");
    }
}; /* end class WrapCallProfiler */

pybind11::dict WrapCallProfiler::result(CallProfiler & profiler)
{
    namespace py = pybind11;

    const RadixTreeNode<CallerProfile> * root = profiler.radix_tree().get_root();
    if (root->empty_children())
    {
        return {};
    }
    py::dict result;
    std::queue<const RadixTreeNode<CallerProfile> *> node_queue;
    std::unordered_map<const RadixTreeNode<CallerProfile> *, py::dict> dict_storage;

    node_queue.push(root);
    dict_storage[root] = result;

    while (!node_queue.empty())
    {
        const RadixTreeNode<CallerProfile> * cur_node = node_queue.front();
        const py::dict & current_dict = dict_storage[cur_node];
        node_queue.pop();

        current_dict["name"] = cur_node->name();
        current_dict["total_time"] = cur_node->data().total_time.count() / 1e6;
        current_dict["count"] = cur_node->data().call_count;
        if (cur_node == profiler.radix_tree().get_current_node())
        {
            current_dict["current_node"] = true;
        }

        py::list children_list;
        for (const auto & child : cur_node->children())
        {
            dict_storage[child.get()] = py::dict();
            py::dict & child_dict = dict_storage[child.get()];
            children_list.append(child_dict);
            node_queue.push(child.get());
        }
        current_dict["children"] = children_list;
    }
    return result;
}

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapCallProfilerProbe : public WrapBase<WrapCallProfilerProbe, CallProfilerProbe>
{
public:
    friend root_base_type;

protected:
    WrapCallProfilerProbe(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](const char * caller_name)
                    {
                        return std::make_unique<CallProfilerProbe>(CallProfiler::instance(), caller_name);
                    }),
                py::arg("caller_name"))
            .def("cancel", &wrapped_type::cancel);
    }
}; /* end class WrapCallProfilerProbe */

void wrap_profile(pybind11::module & mod)
{
    WrapWrapperProfilerStatus::commit(mod, "WrapperProfilerStatus", "WrapperProfilerStatus");
    WrapStopWatch::commit(mod, "StopWatch", "StopWatch");
    WrapTimedEntry::commit(mod, "TimedEntry", "TimeEntry");
    WrapTimeRegistry::commit(mod, "TimeRegistry", "TimeRegistry");
    WrapCallProfiler::commit(mod, "CallProfiler", "CallProfiler");
    WrapCallProfilerProbe::commit(mod, "CallProfilerProbe", "CallProfilerProbe");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: