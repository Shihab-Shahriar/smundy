#ifndef MATH_H
#define MATH_H

#include <cstddef>
#include <initializer_list>
#include <iostream>

#include <Kokkos_Core.hpp>


template <typename T, size_t size>
class Vec
{
public:
    KOKKOS_INLINE_FUNCTION Vec() {
        for(int i=0;i<size;i++) data[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION Vec(T arr[size]){
        for(int i=0;i<size;i++) data[i] = arr[i];
    }

    KOKKOS_INLINE_FUNCTION Vec(std::initializer_list<T> a_args){
        int i=0;
        for(T x: a_args) data[i++] = x;
    }

    KOKKOS_INLINE_FUNCTION T& operator[](int i){
        return data[i];
    }

    KOKKOS_INLINE_FUNCTION const T& operator[](int i) const {
        return data[i];
    }

    KOKKOS_INLINE_FUNCTION Vec operator+(Vec& other){
        T vals[size];
        int i=0;
        for(T x: other.data) {
            vals[i] = x+data[i];
            i++;
        }
        return Vec<T,size>(vals);
    }

    KOKKOS_INLINE_FUNCTION Vec operator+(const Vec& other) const {
        T vals[size];
        int i=0;
        for(T x: other.data) {
            vals[i] = x+data[i];
            i++;
        }
        return Vec<T,size>(vals);
    }

    KOKKOS_INLINE_FUNCTION Vec operator-(const Vec& other) const {
        T vals[size];
        int i=0;
        for(T x: other.data) {
            vals[i] = data[i]-x;
            i++;
        }
        return Vec<T,size>(vals);
    }

    T data[size];
};

template <typename T, size_t size>
std::ostream& operator<<(std::ostream& os, const Vec<T,size> & v){
    os<<"Vec: ("<<v.data[0];
    for(int i=1;i<size;i++) os<<","<<v.data[i];
    return os<<")";
}


#endif


