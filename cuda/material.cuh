#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.cuh"
#include "ray.cuh"
#include "color.cuh"

class material {
  public:
    // virtual ~material() = default;

    __device__ virtual bool scatter( const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const {
        return false;
    }
};

class lambertian : public material {
  public:
    __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
        auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

  private:
    color albedo;
};

class metal : public material {
  public:
    __device__ metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(local_rand_state));
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.f);
    }

  private:
    color albedo;
    double fuzz;
};

#endif