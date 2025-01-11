#pragma once

namespace modmesh {

namespace detail {

template <typename T>
static void swap(T &a, T &b){
    if (a == b) {
        return;
    }
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T>
static int compare(void *a, void *b){
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, 
                    "T must be integral or floating-point type");

    if (a == nullptr || b == nullptr) {
        throw std::invalid_argument(Formatter() << "Null pointer shouldn't be sent into compare function");
    }
    if (a == b) {
        return 0;
    }
    return *static_cast<T *>(a) - *static_cast<T *>(b);
}

template <typename T>
void qsort(T *begin, T *end, int (*cmp)(void *, void *) = compare<T>) {
    ssize_t N = end - begin;
    if (N < 2) {
        return;
    }

    T *end_pos = end - 1;
    T *cur = begin + 1;
    T *pivot = begin;

    while (cur <= end_pos) {
        if (cmp(cur, pivot) < 0) {
            cur++;
        } else {
            swap(*cur, *end_pos);
            end_pos--;
        }
    }
    swap(*pivot, *end_pos);
    pivot = end_pos;

    qsort(begin, pivot, cmp);
    qsort(pivot + 1, end, cmp);
}

} /* end namespace detail */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
