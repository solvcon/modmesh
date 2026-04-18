/*
 * Copyright (c) 2025, Liu, An-Chi <phy.tiger@gmail.com>
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

// Intended to use deprecated Formatter as the demo of comparison with std::format
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <modmesh/base.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <format>
#include <string>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(Formatter, basic_usage)
{
    modmesh::Formatter formatter;
    formatter << "Hello, "
              << "World!"
              << " " << 42;
    std::string formatter_result = formatter.str();

    std::string format_result = std::format("Hello, {}! {}", "World", 42);

    EXPECT_EQ(formatter_result, "Hello, World! 42");
    EXPECT_EQ(format_result, "Hello, World! 42");
    EXPECT_EQ(formatter_result, format_result);
}

TEST(Formatter, numeric_types)
{
    modmesh::Formatter formatter;
    formatter << "int: " << 123 << ", double: " << 3.14159 << ", bool: " << true;
    std::string formatter_result = formatter.str();

    std::string format_result = std::format("int: {}, double: {}, bool: {}", 123, 3.14159, true);

    EXPECT_EQ(formatter_result, "int: 123, double: 3.14159, bool: 1");
    EXPECT_EQ(format_result, "int: 123, double: 3.14159, bool: true");
}

TEST(Formatter, chaining)
{
    std::string formatter_result = (modmesh::Formatter() << "x = " << 10 << ", y = " << 20).str();

    std::string format_result = std::format("x = {}, y = {}", 10, 20);

    EXPECT_EQ(formatter_result, "x = 10, y = 20");
    EXPECT_EQ(format_result, "x = 10, y = 20");
    EXPECT_EQ(formatter_result, format_result);
}

TEST(Formatter, conversion_operator)
{
    std::string formatter_result = modmesh::Formatter() << "Test " << 123;

    std::string format_result = std::format("Test {}", 123);

    EXPECT_EQ(formatter_result, "Test 123");
    EXPECT_EQ(format_result, "Test 123");
    EXPECT_EQ(formatter_result, format_result);
}

TEST(StdFormat, basic_usage)
{
    std::string format_result = std::format("Hello, {}! {}", "World", 42);

    std::string formatter_result = (modmesh::Formatter() << "Hello, "
                                                         << "World!"
                                                         << " " << 42)
                                       .str();

    EXPECT_EQ(format_result, "Hello, World! 42");
    EXPECT_EQ(formatter_result, "Hello, World! 42");
    EXPECT_EQ(format_result, formatter_result);
}

TEST(StdFormat, numeric_types)
{
    std::string format_result = std::format("int: {}, double: {}, bool: {}", 123, 3.14159, true);

    modmesh::Formatter formatter;
    formatter << "int: " << 123 << ", double: " << 3.14159 << ", bool: " << true;
    std::string formatter_result = formatter.str();

    EXPECT_EQ(format_result, "int: 123, double: 3.14159, bool: true");
    EXPECT_EQ(formatter_result, "int: 123, double: 3.14159, bool: 1");
}

TEST(StdFormat, formatting_options)
{
    std::string format_result = std::format("hex: {:#x}, precision: {:.2f}", 255, 3.14159);

    std::string formatter_result = (modmesh::Formatter() << "hex: " << 255 << ", precision: " << 3.14159).str();

    EXPECT_EQ(format_result, "hex: 0xff, precision: 3.14");
    EXPECT_EQ(formatter_result, "hex: 255, precision: 3.14159");
}

TEST(FormatterVsStdFormat, simple_string_comparison)
{
    std::string formatter_result = (modmesh::Formatter() << "Value: " << 42).str();
    std::string format_result = std::format("Value: {}", 42);

    EXPECT_EQ(formatter_result, format_result);
}

TEST(FormatterVsStdFormat, multiple_values_comparison)
{
    std::string formatter_result = (modmesh::Formatter() << "x = " << 10 << ", y = " << 20 << ", z = " << 30).str();
    std::string format_result = std::format("x = {}, y = {}, z = {}", 10, 20, 30);

    EXPECT_EQ(formatter_result, format_result);
}

TEST(FormatterVsStdFormat, performance_formatter)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string result;
    for (int32_t i = 0; i < 10000; ++i)
    {
        result = (modmesh::Formatter() << "Iteration: " << i << ", Value: " << i * 2).str();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Formatter time: " << duration.count() << " microseconds" << std::endl;

    EXPECT_FALSE(result.empty());
}

TEST(FormatterVsStdFormat, performance_std_format)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string result;
    for (int32_t i = 0; i < 10000; ++i)
    {
        result = std::format("Iteration: {}, Value: {}", i, i * 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "std::format time: " << duration.count() << " microseconds" << std::endl;

    EXPECT_FALSE(result.empty());
}

#pragma GCC diagnostic pop

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
