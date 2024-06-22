#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/base.hpp>
#include <modmesh/buffer/BufferBase.hpp>
#include <modmesh/buffer/small_vector.hpp>

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace modmesh
{

namespace detail
{

// Take the remover and deleter classes outside ConcreteBuffer to work around
// https://bugzilla.redhat.com/show_bug.cgi?id=1569374

/**
 * The base class of memory deallocator for ConcreteBuffer.  When the object
 * exists in ConcreteBufferDataDeleter (the unique_ptr deleter), the deleter
 * calls it to release the memory of the ConcreteBuffer data buffer.
 */
struct ConcreteBufferRemover
{

    ConcreteBufferRemover() = default;
    ConcreteBufferRemover(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover(ConcreteBufferRemover &&) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover &&) = default;
    virtual ~ConcreteBufferRemover() = default;

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    virtual void operator()(int8_t * p) const
    {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        delete[] p;
    }

}; /* end struct ConcreteBufferRemover */

struct ConcreteBufferNoRemove : public ConcreteBufferRemover
{

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t *) const override {}

}; /* end struct ConcreteBufferNoRemove */

struct ConcreteBufferDataDeleter
{

    using remover_type = ConcreteBufferRemover;

    ConcreteBufferDataDeleter(ConcreteBufferDataDeleter const &) = delete;
    ConcreteBufferDataDeleter & operator=(ConcreteBufferDataDeleter const &) = delete;

    ConcreteBufferDataDeleter() = default;
    ConcreteBufferDataDeleter(ConcreteBufferDataDeleter &&) = default;
    ConcreteBufferDataDeleter & operator=(ConcreteBufferDataDeleter &&) = default;
    ~ConcreteBufferDataDeleter() = default;
    explicit ConcreteBufferDataDeleter(std::unique_ptr<remover_type> && remover_in)
        : remover(std::move(remover_in))
    {
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t * p) const
    {
        if (!remover)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            delete[] p;
        }
        else
        {
            (*remover)(p);
        }
    }

    std::unique_ptr<remover_type> remover{nullptr};

}; /* end struct ConcreteBufferDataDeleter */

} /* end namespace detail */

/**
 * Untyped and unresizeable memory buffer for contiguous data storage.
 */
class ConcreteBuffer
    : public std::enable_shared_from_this<ConcreteBuffer>
    , public BufferBase<ConcreteBuffer>
{

private:

    struct ctor_passkey
    {
    };

    using data_deleter_type = detail::ConcreteBufferDataDeleter;

public:

    using remover_type = detail::ConcreteBufferRemover;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    using unique_ptr_type = std::unique_ptr<int8_t, data_deleter_type>;

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, ctor_passkey());
    }

    /*
     * This factory method is dangerous since the data pointer passed in will
     * not be owned by the ConcreteBuffer created.  It is an error if the
     * number of bytes of the externally owned buffer doesn't match the value
     * passed in (but we cannot know here).
     */
    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, data, std::move(remover), ctor_passkey());
    }

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, void * data, std::unique_ptr<remover_type> && remover)
    {
        return construct(nbytes, static_cast<int8_t *>(data), std::move(remover));
    }

    static std::shared_ptr<ConcreteBuffer> construct() { return construct(0); }

    std::shared_ptr<ConcreteBuffer> clone() const;

    /**
     * \param[in] nbytes
     *      Size of the memory buffer in bytes.
     */
    ConcreteBuffer(size_t nbytes, const ctor_passkey &);

    /**
     * \param[in] nbytes
     *      Size of the memory buffer in bytes.
     * \param[in] data
     *      Pointer to the memory buffer that is not supposed to be owned by
     *      this ConcreteBuffer.
     * \param[in] remover
     *      The memory deallocator for the unowned data buffer passed in.
     */
    // NOLINTNEXTLINE(readability-non-const-parameter)
    ConcreteBuffer(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, const ctor_passkey &);

    ~ConcreteBuffer() = default;

    ConcreteBuffer() = delete;
    ConcreteBuffer(ConcreteBuffer &&) = delete;
    ConcreteBuffer & operator=(ConcreteBuffer &&) = delete;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#endif
    // Avoid enabled_shared_from_this copy constructor
    // NOLINTNEXTLINE(bugprone-copy-constructor-init)
    ConcreteBuffer(ConcreteBuffer const & other);

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    ConcreteBuffer & operator=(ConcreteBuffer const & other);

    bool has_remover() const noexcept { return bool(m_data.get_deleter().remover); }
    remover_type const & get_remover() const { return *m_data.get_deleter().remover; }
    remover_type & get_remover() { return *m_data.get_deleter().remover; }
    static constexpr const char * name() { return "ConcreteBuffer"; }

private:
    static unique_ptr_type allocate(size_t nbytes);

    size_t m_nbytes;
    unique_ptr_type m_data;
}; /* end class ConcreteBuffer */

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
