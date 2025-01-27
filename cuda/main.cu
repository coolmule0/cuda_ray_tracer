#include <iostream>
#include <time.h>

#include "rtweekend.cuh"

#include "hittable_list.cuh"
#include "sphere.cuh"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color ray_color(const ray& r, hittable **world) {
    hit_record rec;
    if ((*world)->hit(r, interval(0, infinity), rec)) {
        return 0.5f * (rec.normal + color(1,1,1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5f*(unit_direction.y() + 1.0);
    return (1.f-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void render(color *fb, int max_x, int max_y,
                        vec3 camera_center, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, hittable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
    auto ray_direction = pixel_center - camera_center;

    ray r(camera_center, ray_direction);
    fb[pixel_index] = ray_color(r, world);
}

__global__ void create_world(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hittable_list(d_list,2);
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    vec3 camera_center = vec3(0,0,0);
    auto plane_dist = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(nx)/ny);
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);
    auto pixel_delta_u = viewport_u / nx;
    auto pixel_delta_v = viewport_v / ny;

    vec3 focal = vec3(0,0,1);
    vec3 horizontal = vec3(1,0,0);
    vec3 vertical = vec3(0,1,0);
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, plane_dist) - viewport_u/2 - viewport_v/2;

    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // World
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    create_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // hittable_list world;

    // world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    // world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb, nx, ny, 
                        camera_center, pixel00_loc, pixel_delta_u, pixel_delta_v, 
                        d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}