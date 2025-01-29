#include <iostream>
#include <time.h>
#include <curand_kernel.h>

#include "rtweekend.cuh"
#include "camera.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "color.cuh"

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

// __device__ color ray_color(const ray& r, hittable_list **world) {
//     hit_record rec;
//     if ((*world)->hit(r, interval(0, infinity), rec)) {
//         return 0.5f * (rec.normal + color(1,1,1));
//     }

//     vec3 unit_direction = unit_vector(r.direction());
//     auto a = 0.5f*(unit_direction.y() + 1.0);
//     return (1.f-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
// }

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);  
}

__global__ void render(color *fb, camera *camera, hittable_list **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= camera->image_width) || (j >= camera->image_height)) return;

    camera->render(fb, world, rand_state);
}

__global__ void create_world(hittable **d_list, hittable_list **d_world, camera *d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hittable_list(d_list,2);
        // d_world->list_size = 2;

        d_camera->image_width = nx;
        d_camera->image_height = ny;
        d_camera->samples_per_pixel = 100;
        d_camera->ray_bounces = 100;
        d_camera->initialize();
    }
}

__global__ void free_world(hittable **d_list, hittable_list **d_world, camera * d_camera) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
    // delete d_camera;
}

int main() {

    // Check and up the stack size per thread
    size_t stackSize;   
    cudaDeviceSetLimit(cudaLimitStackSize, 8192); //8kb
    // cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);

    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // World & Camera
    camera *d_camera;
    checkCudaErrors(cudaMallocManaged((void **)&d_camera, sizeof(camera *)));
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
    hittable_list **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable_list *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Random numbers per pixel
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    render<<<blocks, threads>>>(fb, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    write_color(std::cout, fb, nx, ny);
    // for (int j = 0; j < ny; j++) {
    //     for (int i = 0; i < nx; i++) {
    //         size_t pixel_index = j*nx + i;
    //         int ir = int(255.99*fb[pixel_index].r());
    //         int ig = int(255.99*fb[pixel_index].g());
    //         int ib = int(255.99*fb[pixel_index].b());
    //         std::cout << ir << " " << ig << " " << ib << "\n";
    //     }
    // }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_camera));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}