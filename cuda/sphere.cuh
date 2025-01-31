#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"
#include "material.cuh"
#include "vec3.cuh"

class sphere : public hittable {
  public:
    __device__ sphere(const point3& center, float radius, material* mat) : center(center), radius(radius), mat(mat) {};

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrt(discriminant)) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrt(discriminant)) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

    __device__ void delete_mat() {
      delete mat;
    }

  private:
    point3 center;
    double radius;
    material *mat;
};

#endif