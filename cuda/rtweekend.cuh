#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

// #define MAXFLOAT infinity
// const float infinity = std::numeric_limits<float>::infinity();
const float infinity = INFINITY;
const float pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.f;
}

// Common Headers

#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

#endif