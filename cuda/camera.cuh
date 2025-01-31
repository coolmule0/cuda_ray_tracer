#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.cuh"
#include "hittable_list.cuh"
#include "interval.cuh"
#include "vec3.cuh"
#include "material.cuh"

class camera {
  public:
    // double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width  = 100;  // Rendered image width in pixel count
    int    image_height = 120;
    int    samples_per_pixel;
    int    ray_bounces;

    __device__ void render(color *fb, hittable_list **world, curandState *rand_state);
    
    __device__ void initialize() {
        // aspect_ratio = float(image_width) / float(image_height);

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = point3(0, 0, 0);

        // Determine viewport dimensions.
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (float(image_width)/image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }


  private:
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    // float  aspect_ratio;   // Ratio of image width over height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below

    __device__ color ray_color(const ray& r, hittable_list **world, curandState *local_rand_state) {
        ray cur_ray = r;
        color cur_attenuation = vec3(1.0,1.0,1.0);
        for(int i = 0; i < ray_bounces; i++) {
            hit_record rec;
            if ((*world)->hit(cur_ray, interval(0.001f, infinity), rec)) {
                ray scattered;
                color attenuation;
                if (rec.mat->scatter(r, rec, attenuation, scattered, local_rand_state)){
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                // vec3 target = rec.p + rec.normal + random_on_hemisphere(rec.normal, local_rand_state);
                // cur_attenuation *= 0.5f;
                // cur_ray = ray(rec.p, target - rec.p);
            }
            else {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                float a = 0.5f*(unit_direction.y() + 1.0f);
                vec3 c = (1.0f-a)*vec3(1.0, 1.0, 1.0) + a*vec3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
            }
        }
        return vec3(0.0,0.0,0.0); // exceeded recursion
    }

    __device__ ray get_ray(int u, int v) {
        auto ray_direction = pixel00_loc + u*pixel_delta_u + v*pixel_delta_v - center;

        return ray(center, ray_direction);
    }
};

__device__ void camera::render(color *fb, hittable_list **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= image_width) || (j >= image_height)) return;
    int pixel_index = j*image_width + i;
    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0,0,0);

    for (int sample=0; sample < samples_per_pixel; sample++) {
        float u = float(i + curand_uniform(&local_rand_state) - 0.5f);
        float v = float(j + curand_uniform(&local_rand_state) - 0.5f);
        auto ray_direction = pixel00_loc + u*pixel_delta_u + v*pixel_delta_v - center;
        ray r = ray(center, ray_direction);

        pixel_color += ray_color(r, world, rand_state);

    }
    fb[pixel_index] = pixel_samples_scale * pixel_color;
}

#endif