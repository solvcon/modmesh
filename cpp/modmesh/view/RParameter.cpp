/*
 * Copyright (c) 2023, Buganini Chiu <buganini@b612.tw>
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

#include <modmesh/view/RParameter.hpp> // Must be the first include.

namespace modmesh
{

static int int64V = 5566;
static double doubleV = 77.88;

int getter_func_i64(int64_t *ptr) {
    return *ptr;
}

void setter_func_i64(int64_t *ptr, int value) {
    *ptr = value;
}

double getter_func_double(double *ptr) {
    return *ptr;
}

void setter_func_double(double *ptr, double value) {
    *ptr = value;
}

enum DataType {
    TYPE_INT64,
    TYPE_DOUBLE,
};

static struct param {
    const char * key;
    void * value;
    int dtype;
} params[] = {
    {"a.b.int64_foo", &int64V, TYPE_INT64},
    {"a.b.double_bar", &doubleV, TYPE_DOUBLE},
};

void openParameterView() {
    pybind11::module pui_module = pybind11::module::import("PUI");
    auto state = pui_module.attr("StateDict")();
    auto paramsList = pybind11::list();

    for(int i=0;i<sizeof(params)/sizeof(params[0]);i++){
        auto key = params[i].key;
        auto ptr = params[i].value;
        switch(params[i].dtype){
            case TYPE_INT64:
                {
                    state[pybind11::str(key)] = getter_func_i64((int64_t *)ptr);
                    auto binding = state(key);
                    binding.attr("bind")(
                        pybind11::cpp_function([ptr](){
                            return getter_func_i64((int64_t *)ptr);
                        }),
                        pybind11::cpp_function([key, ptr](int64_t value){
                            std::cout << "Set " << key << " = " << value << std::endl;
                            setter_func_i64((int64_t *)ptr, value);
                        })
                    );
                    paramsList.append(binding);
                }
                break;
            case TYPE_DOUBLE:
                {
                    state[pybind11::str(key)] = getter_func_double((double *)ptr);
                    auto binding = state(key);
                    binding.attr("bind")(
                        pybind11::cpp_function([ptr](){
                            return getter_func_double((double *)ptr);
                        }),
                        pybind11::cpp_function([key, ptr](double value){
                            std::cout << "Set " << key << " = " << value << std::endl;
                            setter_func_double((double *)ptr, value);
                        })
                    );
                    paramsList.append(binding);
                }
                break;
        }
    }

    pybind11::module params_module = pybind11::module::import("modmesh.params");
    params_module.attr("openParameterView")(paramsList);
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
