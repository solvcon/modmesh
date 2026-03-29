#include <modmesh/modmesh.hpp>
#include <modmesh/transform/transform.hpp>
#include <random>
#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

template <typename T, bool IsPow2>
struct FFTTestParams
{
    using Type = T;
    static constexpr bool is_pow_2 = IsPow2;
};

template <typename TestParam>
class ParsevalTest : public ::testing::Test
{
public:
    using T = typename TestParam::Type;
    static constexpr bool is_pow_2 = TestParam::is_pow_2;

protected:
    static constexpr size_t VN = is_pow_2 ? 1024 : 1000;

    modmesh::SimpleArray<modmesh::Complex<T>> signal{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> out{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};

    // Set up the test fixture: generate the signal once
    void SetUp() override
    {
        std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<T> val_dist{-1.0, 1.0};
        for (unsigned int i = 0; i < VN; ++i)
        {
            T val = val_dist(rng);
            signal[i] = modmesh::Complex<T>{val, 0.0};
        }
    }

    // Function to verify energy conservation (Parseval's theorem)
    void verify_parseval()
    {
        T psd_sig = 0.0, psd_out = 0.0;

        for (unsigned int i = 0; i < VN; ++i)
        {
            psd_sig += signal[i].norm();
            psd_out += out[i].norm() / VN;
        }

        // TODO: When FFT / DFT uses float, the accumulation error is around 1e-3,
        // but when using double, the accumulation error is around 1e-12.
        // We need to investigate the root cause to determine whether this is an issue
        // or if there is a way to optimize it.
        // Expect the total energy in the time and frequency domains to be equal
        EXPECT_NEAR(psd_sig, psd_out, (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
    }
};

template <typename TestParam>
class DeltaFunctionTest : public ::testing::Test
{
public:
    using T = typename TestParam::Type;
    static constexpr bool is_pow_2 = TestParam::is_pow_2;

protected:
    static constexpr size_t VN = is_pow_2 ? 1024 : 1000;

    std::mt19937 rng{std::random_device{}()};

    modmesh::SimpleArray<modmesh::Complex<T>> signal{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> out{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};

    void SetUp() override
    {
        signal[0] = modmesh::Complex<T>{1.0, 0.0};
    }

    void verify_delta_function()
    {
        T expected_mag = static_cast<T>(1.0);

        for (unsigned int i = 0; i < VN; ++i)
        {
            T mag = out[i].norm();
            EXPECT_NEAR(mag, expected_mag, (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
        }
    }
};

template <typename TestParam>
class InverseTest : public ::testing::Test
{
public:
    using T = typename TestParam::Type;
    static constexpr bool is_pow_2 = TestParam::is_pow_2;

protected:
    static constexpr size_t VN = is_pow_2 ? 1024 : 1000;

    std::mt19937 rng{std::random_device{}()};

    modmesh::SimpleArray<modmesh::Complex<T>> signal{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> freq_domain{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> time_domain{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};

    void SetUp() override
    {
        std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<T> val_dist{-1.0, 1.0};
        for (unsigned int i = 0; i < VN; ++i)
        {
            T val = val_dist(rng);
            signal[i] = modmesh::Complex<T>{val, 0.0};
        }
    }

    void verify_inverse_fft_function()
    {
        for (unsigned int i = 0; i < VN; ++i)
        {
            EXPECT_NEAR(signal[i].real(), time_domain[i].real(), (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
            EXPECT_NEAR(signal[i].imag(), time_domain[i].imag(), (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
        }
    }
};

typedef ::testing::Types<
    FFTTestParams<float, true>,
    FFTTestParams<float, false>,
    FFTTestParams<double, true>,
    FFTTestParams<double, false>>
    TestTypes;

TYPED_TEST_SUITE(ParsevalTest, TestTypes);
TYPED_TEST_SUITE(DeltaFunctionTest, TestTypes);
TYPED_TEST_SUITE(InverseTest, TestTypes);

TYPED_TEST(ParsevalTest, fft)
{
    modmesh::FourierTransform::fft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->out, "cpu");
    this->verify_parseval();
#if defined(BUILD_CUDA)
    modmesh::FourierTransform::fft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->out, "cuda");
    this->verify_parseval();
#endif
}

TYPED_TEST(ParsevalTest, dft)
{
    modmesh::FourierTransform::dft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->out);

    this->verify_parseval();
}

TYPED_TEST(DeltaFunctionTest, fft)
{
    modmesh::FourierTransform::fft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->out, "cpu");
    this->verify_delta_function();
#if defined(BUILD_CUDA)
    modmesh::FourierTransform::fft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->out, "cuda");
    this->verify_delta_function();
#endif
}

TYPED_TEST(DeltaFunctionTest, dft)
{
    modmesh::FourierTransform::dft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->out);

    this->verify_delta_function();
}

TYPED_TEST(InverseTest, fft)
{
    modmesh::FourierTransform::fft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->freq_domain, "cpu");
    modmesh::FourierTransform::ifft<modmesh::Complex, typename TypeParam::Type>(this->freq_domain, this->time_domain, "cpu");
    this->verify_inverse_fft_function();
#if defined(BUILD_CUDA)
    modmesh::FourierTransform::fft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->freq_domain, "cuda");
    modmesh::FourierTransform::ifft<modmesh::Complex, typename TypeParam::Type>(this->freq_domain, this->time_domain, "cuda");
    this->verify_inverse_fft_function();
#endif
}

TYPED_TEST(InverseTest, dft)
{
    modmesh::FourierTransform::dft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->freq_domain);
    modmesh::FourierTransform::ifft<modmesh::Complex, typename TypeParam::Type>(this->freq_domain, this->time_domain, "cpu");
    this->verify_inverse_fft_function();
#if defined(BUILD_CUDA)
    // output buffer must be zero-initialized before calling dft
    for (size_t i = 0; i < this->freq_domain.size(); ++i)
        this->freq_domain[i] = {0.0, 0.0};
    modmesh::FourierTransform::dft<modmesh::Complex, typename TypeParam::Type>(this->signal, this->freq_domain);
    modmesh::FourierTransform::ifft<modmesh::Complex, typename TypeParam::Type>(this->freq_domain, this->time_domain, "cuda");
    this->verify_inverse_fft_function();
#endif
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
