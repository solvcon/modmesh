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

    static void deallocate_memory(int8_t * p, size_t alignment)
    {
        if (alignment > 0)
        {
#ifdef _WIN32
            _aligned_free(p);
#else
            std::free(p);
#endif
        }
        else
        {
            std::free(p);
        }
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    virtual void operator()(int8_t * p, size_t alignment) const
    {
        deallocate_memory(p, alignment);
    }

}; /* end struct ConcreteBufferRemover */

struct ConcreteBufferNoRemove : public ConcreteBufferRemover
{

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t *, size_t) const override {}

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
    explicit ConcreteBufferDataDeleter(std::unique_ptr<remover_type> && remover_in, size_t alignment_in = 0)
        : remover(std::move(remover_in))
        , alignment(alignment_in)
    {
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t * p) const
    {
        if (!remover)
        {
            remover_type::deallocate_memory(p, alignment);
        }
        else
        {
            (*remover)(p, alignment);
        }
    }

    std::unique_ptr<remover_type> remover{nullptr};
    size_t alignment = 0; // Alignment of the data buffer in bytes. 0 means no alignment.

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
    using size_type = std::size_t;

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, size_t alignment = 0)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, alignment, ctor_passkey());
    }

    /*
     * This factory method is dangerous since the data pointer passed in will
     * not be owned by the ConcreteBuffer created.  It is an error if the
     * number of bytes of the externally owned buffer doesn't match the value
     * passed in (but we cannot know here).
     */
    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, size_t alignment = 0)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, data, std::move(remover), alignment, ctor_passkey());
    }

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, void * data, std::unique_ptr<remover_type> && remover, size_t alignment = 0)
    {
        return construct(nbytes, static_cast<int8_t *>(data), std::move(remover), alignment);
    }

    /// Construct an empty ConcreteBuffer with no data and no alignment.
    static std::shared_ptr<ConcreteBuffer> construct() { return construct(0, 0); }

    std::shared_ptr<ConcreteBuffer> clone() const
    {
        std::shared_ptr<ConcreteBuffer> ret = construct(nbytes(), m_alignment);
        std::copy_n(data(), size(), (*ret).data());
        return ret;
    }

    /**
     * \param[in] nbytes
     *      Size of the memory buffer in bytes.
     * \param[in] alignment
     *      Alignment for the memory buffer in bytes.
     *      0 means no alignment. Valid values are 0, 16, 32, or 64.
     */
    ConcreteBuffer(size_t nbytes, size_t alignment, const ctor_passkey &)
        : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(nbytes)
        , m_alignment(validate_alignment(alignment, "ConcreteBuffer::ConcreteBuffer"))
        , m_data(allocate(nbytes, m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
    }

    /**
     * \param[in] nbytes
     *      Size of the memory buffer in bytes.
     * \param[in] data
     *      Pointer to the memory buffer that is not supposed to be owned by
     *      this ConcreteBuffer.
     * \param[in] remover
     *      The memory deallocator for the unowned data buffer passed in.
     * \param[in] alignment
     *      Alignment for the memory buffer in bytes.
     *      0 means no alignment. Valid values are 0, 16, 32, or 64.
     */
    // NOLINTNEXTLINE(readability-non-const-parameter)
    ConcreteBuffer(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, size_t alignment, const ctor_passkey &)
        : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(nbytes)
        , m_alignment(validate_alignment(alignment, "ConcreteBuffer::ConcreteBuffer"))
        , m_data(data, data_deleter_type(std::move(remover), m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
    }

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
    ConcreteBuffer(ConcreteBuffer const & other)
        : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(other.m_nbytes)
        , m_alignment(other.m_alignment)
        , m_data(allocate(other.m_nbytes, other.m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
        if (size() != other.size())
        {
            throw std::out_of_range("Buffer size mismatch");
        }
        std::copy_n(other.data(), size(), data());
    }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    ConcreteBuffer & operator=(ConcreteBuffer const & other)
    {
        if (this != &other)
        {
            if (size() != other.size())
            {
                throw std::out_of_range("Buffer size mismatch");
            }
            std::copy_n(other.data(), size(), data());
        }
        return *this;
    }

    bool has_remover() const noexcept { return bool(m_data.get_deleter().remover); }
    remover_type const & get_remover() const { return *m_data.get_deleter().remover; }
    remover_type & get_remover() { return *m_data.get_deleter().remover; }

    size_type alignment() const noexcept { return m_alignment; }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    using unique_ptr_type = std::unique_ptr<int8_t, data_deleter_type>;

    static constexpr const char * name() { return "ConcreteBuffer"; }

private:
    static unique_ptr_type allocate(size_t nbytes, size_t alignment)
    {
        unique_ptr_type ret(nullptr, data_deleter_type());
        if (0 != nbytes)
        {
            void * ptr = nullptr;
            if (alignment > 0)
            {
                validate_size_alignment(nbytes, alignment, "ConcreteBuffer::allocate");
#ifdef _WIN32
                ptr = _aligned_malloc(nbytes, alignment);
#else
                ptr = std::aligned_alloc(alignment, nbytes);
#endif
            }
            else
            {
                ptr = std::malloc(nbytes);
            }
            if (!ptr)
            {
                throw std::bad_alloc();
            }
            ret = unique_ptr_type(static_cast<int8_t *>(ptr), data_deleter_type(nullptr, alignment));
        }
        return ret;
    }

    size_t m_nbytes;
    size_t m_alignment = 0; // Alignment of the data buffer in bytes. 0 means no alignment.
    unique_ptr_type m_data;
}; /* end class ConcreteBuffer */

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
