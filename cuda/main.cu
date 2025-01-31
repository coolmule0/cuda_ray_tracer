#include <iostream>
#include <time.h>
#include <curand_kernel.h>

#include "rtweekend.cuh"
#include "camera.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "color.cuh"
#include "material.cuh"

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

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable **d_list, hittable_list **d_world, camera *d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        d_camera->image_width = nx;
        d_camera->image_height = ny;
        d_camera->samples_per_pixel = 100;
        d_camera->ray_bounces = 100;
        d_camera->vfov = 20;
        d_camera->lookfrom = point3(13,2,3);
        d_camera->lookat   = point3(0,0,0);
        d_camera->vup      = vec3(0,1,0);
        d_camera->initialize();

        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                                new lambertian(vec3(0.5, 0.5, 0.5)));

        int i=1;
        for (int a = 0; a < 6; a++) {
            for (int b = 0; b < 6; b++) {
                auto choose_mat = RND;
                point3 center(a + 0.9*RND, 0.2, b + 0.9*RND);

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    // shared_ptr<material> sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        d_list[i++] = new sphere(center, 0.2, new lambertian(color(vec3(RND*RND, RND*RND, RND*RND))));
                        // auto albedo = color::random() * color::random();
                        // sphere_material = make_shared<lambertian>(albedo);
                        // world.add(make_shared<sphere>(center, 0.2, sphere_material));
                    } else {
                        // metal
                        d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));           
                        // auto albedo = color::random(0.5, 1);
                        // auto fuzz = random_double(0, 0.5);
                        // sphere_material = make_shared<metal>(albedo, fuzz);
                        // world.add(make_shared<sphere>(center, 0.2, sphere_material));
                    }
                }
            }
        }

        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 6*6+1+2);
    }
}

__global__ void free_world(hittable **d_list, hittable_list **d_world, camera * d_camera) {
    for(int i=0; i < 6*6+1+2; i++) {
        ((sphere *)d_list[i])->delete_mat();
        delete d_list[i];
    }
    delete *d_world;
    // delete d_camera;
}

int main() {

    // Check and up the stack size per thread
    size_t stackSize;   
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8192*16)); //8kb
    // cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);


    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    // int blockSize;   // The launch configurator returned block size 
    // int minGridSize; // The minimum grid size needed to achieve the 
    //                // maximum occupancy for a full device launch 
    // int gridSize;    // The actual grid size needed, based on input size 

    // cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
    //                                   render, 0, 0); 

    // // Round up according to array size 
    // gridSize = (nx * ny + blockSize - 1) / blockSize;
    // std::cerr << gridSize << " blocks and " << blockSize << " threads\n";

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Random numbers per pixel
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // World & Camera
    camera *d_camera;
    checkCudaErrors(cudaMallocManaged((void **)&d_camera, sizeof(camera *)));
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
    hittable_list **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable_list *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state);
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