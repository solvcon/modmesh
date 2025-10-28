#pragma once

/*
 * Copyright (c) 2025, Chun-Shih Chang <austin20463@gmail.com>
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

#include <modmesh/linalg/factorization.hpp>
#include <modmesh/math/math.hpp>

namespace modmesh
{

namespace detail
{
template <typename T>
struct select_real_t;
} /* end namespace detail */

/**
 * Reference: FilterPy KalmanFilter documentation
 * https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
 */
template <typename T>
class KalmanFilter
{

public:

    using value_type = T;
    using real_type = typename detail::select_real_t<value_type>::type;
    using array_type = SimpleArray<T>;

private:

    size_t m_state_size; // state dimension
    size_t m_measurement_size; // measurement dimension
    size_t m_control_size; // control dimension

    array_type m_f; // state transition matrix (state_sizexstate_size)
    array_type m_q; // process noise covariance (state_sizexstate_size)
    array_type m_h; // measurement matrix (measurement_sizexstate_size)
    array_type m_r; // measurement noise covariance (measurement_sizexmeasurement_size)
    array_type m_p; // state covariance (state_sizexstate_size)
    array_type m_b; // control matrix (state_sizexcontrol_size)
    array_type m_x; // state vector (state_sizex1)
    array_type m_i; // identity matrix (state_sizexstate_size)

    real_type m_jitter; // regularization jitter for numerical stability

public:

    /**
     * @brief Construct a Kalman filter.
     *
     * @details
     * Creates a Kalman filter with the specified system matrices and noise
     * parameters. The process and measurement noise covariances are created
     * from scalar variance values (multiplied by identity matrices). The
     * initial state covariance is set to the identity matrix.
     *
     * Control input: an external force or command applied to the system
     * (e.g., thrust in aircraft, motor command in robot). If the control
     * matrix B is empty (size 0), control input u is disabled. See
     * https://kalmanfilter.net/multiExamples.html for examples.
     *
     * @param x Initial state vector.
     * @param f State transition matrix F.
     * @param b Control matrix B (empty to disable control input u).
     * @param h Measurement matrix H.
     * @param process_noise Process noise standard deviation.
     * @param measurement_noise Measurement noise standard deviation.
     * @param jitter Numerical stability jitter.
     */
    KalmanFilter(
        array_type const & x,
        array_type const & f,
        array_type const & b,
        array_type const & h,
        real_type process_noise,
        real_type measurement_noise,
        real_type jitter)
        : m_state_size(x.shape(0))
        , m_measurement_size(h.shape(0))
        , m_control_size((b.ndim() == 2) ? b.shape(1) : 0)
        , m_f(f)
        , m_q(array_type::scaled_eye(m_state_size, static_cast<T>(process_noise * process_noise)))
        , m_h(h)
        , m_r(array_type::scaled_eye(m_measurement_size, static_cast<T>(measurement_noise * measurement_noise)))
        , m_p(array_type::eye(m_state_size))
        , m_b(b)
        , m_x(x)
        , m_i(array_type::eye(m_state_size))
        , m_jitter(jitter)
    {
        check_dimensions();
    }

    array_type const & state() const { return m_x; }
    array_type const & covariance() const { return m_p; }

    /**
     * @brief Predict step without control input u.
     *
     * @details
     * Advances the state using only the transition model and process noise.
     * Represents a passive system with no external actuation.
     *
     * @note Common in finance, tracking, or passive simulation tasks.
     *
     * @see predict(const array_type&), update(const array_type&)
     */
    void predict()
    {
        // x <- F x  (m_x <- m_f @ m_x)
        predict_state();

        // P <- F P F^H + Q  (m_p <- m_f @ m_p @ m_f^H + m_q)
        predict_covariance();

        // P <- 0.5(P + P^H)  (m_p <- 0.5(m_p + m_p^H))
        m_p = m_p.symmetrize();
    }

    /**
     * @brief Predict step with control input u.
     *
     * @details
     * Updates the state while applying a known control SimpleArray.
     * The control term affects only the state mean.
     *
     * @note Example: in a vehicle model, the state may contain position and velocity,
     * while the control input represents acceleration (throttle or brake). The
     * control term updates how the state evolves between sensor measurements.
     *
     * @param u Control input.
     *
     * @see predict(), update(const array_type&)
     */
    void predict(array_type const & u)
    {
        check_control(u);

        // x <- F x + B u  (m_x <- m_f @ m_x + m_b @ u)
        predict_state(u);

        // P <- F P F^H + Q  (m_p <- m_f @ m_p @ m_f^H + m_q)
        predict_covariance();

        // P <- 0.5(P + P^H)  (m_p <- 0.5(m_p + m_p^H))
        m_p = m_p.symmetrize();
    }

    /**
     * @brief Update step (measurement correction).
     *
     * @details
     * Incorporates a new measurement to refine the predicted state and covariance
     * using the Kalman gain.
     *
     * @param z Measurement input.
     *
     * @see predict(), predict(const array_type&)
     */
    void update(array_type const & z)
    {
        check_measurement(z);

        // y <- z - H x  (y <- z - m_h @ m_x)
        array_type y = innovation(z);

        // S <- H P H^H + R + jitter I  (s <- m_h @ m_p @ m_h^H + m_r + m_jitter @ I)
        array_type s = innovation_covariance();

        // K <- P H^H S^{-1} via LLT solve  (k <- m_p @ m_h^H @ s^{-1})
        array_type k = kalman_gain(s);

        // x <- x + K y  (m_x <- m_x + k @ y)
        update_state(k, y);

        // P <- (I-K H) P (I-K H)^H + K R K^H  (m_p <- (I-k@m_h) @ m_p @ (I-k@m_h)^H + k @ m_r @ k^H)
        update_covariance(k);
    }

private:

    void check_dimensions();
    void check_measurement(array_type const & z);
    void check_control(array_type const & u);

    // Predict
    void predict_state();
    void predict_state(array_type const & u);
    void predict_covariance();

    // Update
    array_type innovation(array_type const & z);
    array_type innovation_covariance();
    array_type kalman_gain(array_type const & s);
    void update_state(array_type const & k, array_type const & y);
    void update_covariance(array_type const & k);

}; /* end class KalmanFilter */

template <typename T>
void KalmanFilter<T>::check_dimensions()
{
    if (m_f.ndim() != 2 || m_f.shape(0) != m_state_size || m_f.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The state transition SimpleArray f must be state_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_f.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_f.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_h.ndim() != 2 || m_h.shape(0) != m_measurement_size || m_h.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The measurement SimpleArray h must be measurement_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_h.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_h.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_q.ndim() != 2 || m_q.shape(0) != m_state_size || m_q.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The process noise covariance SimpleArray q must be state_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_q.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_q.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_r.ndim() != 2 || m_r.shape(0) != m_measurement_size || m_r.shape(1) != m_measurement_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The measurement noise covariance SimpleArray r must be measurement_sizexmeasurement_size, but got shape (";
        for (ssize_t i = 0; i < m_r.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_r.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_p.ndim() != 2 || m_p.shape(0) != m_state_size || m_p.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The state covariance SimpleArray p must be state_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_p.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_p.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_x.ndim() != 1 || m_x.shape(0) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The state SimpleArray x must be 1D of length state_size, but got shape (";
        for (ssize_t i = 0; i < m_x.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_x.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }

    if (m_control_size > 0)
    {
        if (m_b.ndim() != 2)
        {
            std::ostringstream oss;
            oss << "KalmanFilter::check_dimensions: The control SimpleArray b must be 2D when control_size > 0, but got " << m_b.ndim() << "D";
            throw std::invalid_argument(oss.str());
        }
        if (m_b.shape(0) != m_state_size || m_b.shape(1) != m_control_size)
        {
            std::ostringstream oss;
            oss << "KalmanFilter::check_dimensions: The control SimpleArray b must be state_sizexcontrol_size, but got shape (";
            for (ssize_t i = 0; i < m_b.ndim(); ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                oss << m_b.shape(i);
            }
            oss << ")";
            throw std::invalid_argument(oss.str());
        }
    }
    else
    {
        if (m_b.ndim() != 2 || m_b.shape(1) != 0)
        {
            std::ostringstream oss;
            oss << "KalmanFilter::check_dimensions: The control SimpleArray b must be state_sizex0 when control_size = 0, but got shape (";
            for (ssize_t i = 0; i < m_b.ndim(); ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                oss << m_b.shape(i);
            }
            oss << ")";
            throw std::invalid_argument(oss.str());
        }
    }
}

template <typename T>
void KalmanFilter<T>::predict_state()
{
    // x <- F x  (m_x <- m_f @ m_x)
    array_type x_col = m_x.reshape(small_vector<size_t>{m_state_size, 1});
    x_col = m_f.matmul(x_col);
    m_x = x_col.reshape(small_vector<size_t>{m_state_size});
}

template <typename T>
void KalmanFilter<T>::predict_state(array_type const & u)
{
    // x <- F x + B u  (m_x <- m_f @ m_x + m_b @ u)
    array_type x_col = m_x.reshape(small_vector<size_t>{m_state_size, 1});
    array_type u_col = u.reshape(small_vector<size_t>{m_control_size, 1});
    x_col = m_f.matmul(x_col).add(m_b.matmul(u_col));
    m_x = x_col.reshape(small_vector<size_t>{m_state_size});
}

template <typename T>
void KalmanFilter<T>::predict_covariance()
{
    // P <- F P F^H + Q  (m_p <- m_f @ m_p @ m_f^H + m_q)
    array_type f_h = m_f.hermitian();
    m_p = m_f.matmul(m_p).matmul(f_h).add(m_q);
}

template <typename T>
void KalmanFilter<T>::check_measurement(array_type const & z)
{
    if (z.ndim() != 1 || z.shape(0) != m_measurement_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_measurement: The measurement SimpleArray z must be 1D of length measurement_size (" << m_measurement_size << "), but got shape (";
        for (ssize_t i = 0; i < z.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << z.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
}

template <typename T>
void KalmanFilter<T>::check_control(array_type const & u)
{
    if (m_control_size == 0)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_control: Control input not supported: control_size is 0";
        throw std::invalid_argument(oss.str());
    }
    if (u.ndim() != 1 || u.shape(0) != m_control_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_control: The control SimpleArray u must be 1D of length control_size (" << m_control_size << "), but got shape (";
        for (ssize_t i = 0; i < u.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << u.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
}

template <typename T>
typename KalmanFilter<T>::array_type KalmanFilter<T>::innovation(array_type const & z)
{
    // y <- z - H x  (y <- z - m_h @ m_x)
    array_type x_col = m_x.reshape(small_vector<size_t>{m_state_size, 1});
    array_type hx = m_h.matmul(x_col);
    array_type hx_vec = hx.reshape(small_vector<size_t>{m_measurement_size});
    return z.sub(hx_vec);
}

template <typename T>
typename KalmanFilter<T>::array_type KalmanFilter<T>::innovation_covariance()
{
    // S <- H P H^H + R + jitter I  (s <- m_h @ m_p @ m_h^H + m_r + m_jitter @ I)
    array_type h_h = m_h.hermitian();
    array_type hph_h = m_h.matmul(m_p).matmul(h_h);
    array_type s = hph_h.add(m_r).add(array_type::scaled_eye(m_measurement_size, static_cast<T>(m_jitter)));

    // S <- 0.5(S + S^H)  (s <- 0.5(s + s^H))
    return s.symmetrize();
}

template <typename T>
typename KalmanFilter<T>::array_type KalmanFilter<T>::kalman_gain(array_type const & s)
{
    // K <- P H^H S^{-1} via LLT solve
    array_type B = m_h.matmul(m_p); // H@P = B
    array_type X = llt_solve(s, B); // S@X = B
    return X.hermitian(); // K = X^H
}

template <typename T>
void KalmanFilter<T>::update_state(array_type const & k, array_type const & y)
{
    // x <- x + K y  (m_x <- m_x + k @ y)
    array_type y_col = y.reshape(small_vector<size_t>{m_measurement_size, 1});
    array_type ky = k.matmul(y_col);
    array_type ky_vec = ky.reshape(small_vector<size_t>{m_state_size});
    m_x = m_x.add(ky_vec);
}

template <typename T>
void KalmanFilter<T>::update_covariance(array_type const & k)
{
    // P <- (I-K H) P (I-K H)^H + K R K^H  (m_p <- (I-k@m_h)@m_p@(I-k@m_h)^H + k@m_r@k^H)
    array_type kh = k.matmul(m_h);
    array_type i_minus_kh = m_i.sub(kh);
    array_type i_minus_kh_h = i_minus_kh.hermitian();
    array_type k_h = k.hermitian();
    array_type term1 = i_minus_kh.matmul(m_p).matmul(i_minus_kh_h);
    array_type term2 = k.matmul(m_r).matmul(k_h);
    m_p = term1.add(term2);

    // P <- 0.5(P + P^H)  (m_p <- 0.5(m_p + m_p^H))
    m_p = m_p.symmetrize();
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
