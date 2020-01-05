#pragma once

/*
 * Copyright (c) 2020, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include "modmesh/small_vector.hpp"

#include <stdexcept>
#include <memory>

namespace modmesh
{

/**
 * Untyped and unresizeable memory buffer for contiguous data storage.
 */
class ConcreteBuffer
  : public std::enable_shared_from_this<ConcreteBuffer>
{

private:

    struct ctor_passkey {};

public:

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, ctor_passkey());
    }

    static std::shared_ptr<ConcreteBuffer> construct() { return construct(0); }

    std::shared_ptr<ConcreteBuffer> clone() const
    {
        std::shared_ptr<ConcreteBuffer> ret = construct(nbytes());
        std::copy_n(data(), size(), (*ret).data());
        return ret;
    }

    /**
     * \param[in] length Memory buffer length.
     */
    ConcreteBuffer(size_t nbytes, const ctor_passkey &)
      : m_nbytes(nbytes)
      , m_data(allocate(nbytes))
    {}

    ~ConcreteBuffer() = default;

    ConcreteBuffer() = delete;
    ConcreteBuffer(ConcreteBuffer const & ) = delete;
    ConcreteBuffer(ConcreteBuffer       &&) = delete;

    ConcreteBuffer & operator=(ConcreteBuffer const & other)
    {
        if (this != &other)
        {
            if (size() != other.size())
            { throw std::out_of_range("Buffer size mismatch"); }
            std::copy_n(other.data(), size(), data());
        }
        return *this;
    }

    ConcreteBuffer & operator=(ConcreteBuffer &&) = delete;

    explicit operator bool() const noexcept { return bool(m_data); }

    size_t nbytes() const noexcept { return m_nbytes; }
    size_t size() const noexcept { return nbytes(); }

    /* Backdoor */
    char const * data() const noexcept { return data<char>(); }
    char       * data()       noexcept { return data<char>(); }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    template<typename T> T const * data() const noexcept { return reinterpret_cast<T*>(m_data.get()); }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    template<typename T> T       * data()       noexcept { return reinterpret_cast<T*>(m_data.get()); }

private:

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    using unique_ptr_type = std::unique_ptr<char, std::default_delete<char[]>>;
    static_assert(sizeof(size_t) == sizeof(unique_ptr_type), "sizeof(Buffer::m_data) must be a word");

    static unique_ptr_type allocate(size_t nbytes)
    {
        unique_ptr_type ret;
        if (0 != nbytes)
        {
            ret = unique_ptr_type(new char[nbytes]);
        }
        else
        {
            ret = unique_ptr_type();
        }
        return ret;
    }

    size_t m_nbytes;
    unique_ptr_type m_data;

}; /* end class ConcreteBuffer */

} /* end namespace modmesh */

/* vim: set et ts=4 sw=4: */
