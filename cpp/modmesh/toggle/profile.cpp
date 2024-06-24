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

#include <modmesh/toggle/profile.hpp>

namespace modmesh
{

std::string TimeRegistry::detailed_report() const
{
    std::ostringstream ostm;
    /// Header
    ostm
        << std::setw(40) << total_call_count()
        << " function calls in " << total_time()
        << " seconds" << std::endl;
    ostm
        << std::endl
        << std::setw(40) << "Function Name"
        << std::setw(25) << "Call Count"
        << std::setw(25) << "Total Time (s)"
        << std::setw(25) << "Per Call (s)"
        << std::setw(25) << "Cumulative Time (s)"
        << std::setw(25) << "Per Call (s)"
        << std::endl;

    /// Body
    for (auto it = m_entry.begin(); it != m_entry.end(); ++it)
    {
        ostm
            << std::setw(40) << it->first
            << std::setw(25) << it->second.count()
            << std::setw(25) << it->second.time()
            << std::setw(25) << it->second.time() / it->second.count()
            << std::setw(25) << it->second.ctime()
            << std::setw(25) << it->second.ctime() / it->second.count()
            << std::endl;
    }
    return ostm.str();
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
