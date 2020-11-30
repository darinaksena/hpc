
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <stdio.h>
#include <cmath>
#include <float.h>
#include <chrono>
#include <iostream>

#define BLOCK_SIZE 16
#define pi 3.1415926535
#define N 5 //p = 5

void fitness(int count, int countInd, double* individuals, double* arrInd) {

    double sumError = 0.0;
    double f_approx;
    double h = pi / count;


    for (int i = 0; i < countInd; i++) {

        for (int j = 0; j < count; j++) {
            f_approx = 0;
            for (int k = 0; k < N; k++) {
                f_approx += individuals[i * N + k] * powf(j * h + h, k);
            }
            //àïïðîêñèìèðóþ sin(x) ïîòîìó, ÷òî 4 ãîäà áàêàëàâðèàòà
            sumError += powf(sumError - sin(j * h + h), 2);
        }

        arrInd[i] = sumError;
        sumError = 0.0;
    }


}

__global__ void fitness_GPU(int count, int countInd, double* individuals, double* arrInd) {
    double sumError = 0.0;
    double f_approx;
    double h = pi / count;
    double X = 0.0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index < countInd) {
        for (int i = index; i < countInd; i += stride) {

            for (int j = 0; j < count; j++) {
                f_approx = 0.0;
                for (int k = 0; k < N; k++) {
                    X = powf(j * h + h, k);
                    f_approx += individuals[i * N + k] * X;
                }
                sumError += powf(sin(j * h + h) - f_approx,2);
            }

            arrInd[i] = sumError;
            sumError = 0.0;
        }
    }
}

void Selection(int* indexes, double* arrInd, int countInd, int countParents, double* bP, double* individuals) {
    for (int i = 0; i < countInd; i++) {
        indexes[i] = i;
    }

    thrust::sort_by_key(arrInd, arrInd + countInd, indexes);

    for (int i = 0; i < countParents; i++) {
        for (int j = 0; j < N; j++) {
            bP[i * N + j] = individuals[N * indexes[i] + j];
        }
    }

}

__global__ void Breeding_GPU(double Em, double Dm, int count, int countInd, double* d_bP, double* d_arrInd, double* individuals) {

    int countParents = 10;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index < countParents) {
        for (int i = index; i < countParents; i += stride) {
            int countIndividuals = countInd / countParents;
            for (int j = 0; j < countIndividuals; j++) {

                curandState state;
                curand_init((unsigned long long)clock() + index, 0, 0, &state);

                int n = floor(curand_uniform_double(&state) * N);


                //Crossover
                for (int k = 0; k < n; k++) {
                    individuals[i * countIndividuals * N + j * N + k] = d_bP[i * N + k];
                }

                for (int k = n; k < N; k++) {
                    individuals[i * countIndividuals * N + j * N + k] = d_bP[(countParents - i) * N + k];
                }
                curand_init((unsigned long long)clock() + index, 0, 0, &state);

                //Mutation
                if (curand_uniform_double(&state) > 0.5) {

                    double d = Dm * curand_uniform_double(&state);
                    double m = Em;
                    if (curand_uniform_double(&state) > 0.5)
                        m += d;
                    else
                        m -= d;
                    int nn = (int)(curand_uniform_double(&state) * N);
                    if (curand_uniform_double(&state) > 0.5)
                        individuals[i * countIndividuals * N + j * N + nn] += m;
                    else
                        individuals[i * countIndividuals * N + j * N + nn] -= m;
                }
            }
        }
    }

}




__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int main()
{

    int count, countInd, countParents, maxIter, maxConstIter;
    double Em, Dm;

    countParents = 10;

    std::cout << "Enter count of points (500 - 1000): " << std::endl;
    std::cin >> count;

    std::cout << "Enter count of individuals (1000 - 2000): " << std::endl;
    std::cin >> countInd;

    std::cout << "Enter mean for Mutation: " << std::endl;
    std::cin >> Em;

    std::cout << "Enter variance for Mutation: " << std::endl;
    std::cin >> Dm;

    std::cout << "Enter max count of generations: " << std::endl;
    std::cin >> maxIter;

    std::cout << "Enter max count of generations with same results: " << std::endl;
    std::cin >> maxConstIter;


    //first generation
    double* h_bP = new double[countParents * N];
    for (int i = 0; i < countParents; i++) {
        for (int j = 0; j < N; j++) {
            h_bP[i * N + j] = 0.0;
        }
    }

    //evolution
    int* h_indexes = new int[countInd];
    double* h_arrInd = new double[countInd];
    double* h_childrens = new double[countInd * N];

    for (int i = 0; i < countInd; i++) {
        for (int j = 0; j < N; j++) {
            h_childrens[i * N + j] = 0.0;
        }
    }


    int* d_indexes;
    double* d_arrInd;
    double* d_bP;
    double* d_childrens;
    double min = DBL_MAX, val = DBL_MAX;
    int indBest, sameIter = 1;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto begin = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    float gpu_elapsed_time_ms;
    float sumTime = 0.0;


    cudaMalloc((void**)&d_arrInd, countInd * sizeof(double));
    cudaMalloc((void**)&d_bP, countParents * N * sizeof(double));
    cudaMalloc((void**)&d_childrens, countInd * N * sizeof(double));
    //cudaMalloc((void**)&d_indexes, countInd * sizeof(int));

    for (int generation = 1; generation <= maxIter; generation++) {


        cudaEventRecord(start, 0);
        cudaMemcpy(d_bP, h_bP, countParents * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!");
        }
        Breeding_GPU << <countParents * countInd, 1 >> > (Em, Dm, count, countInd, d_bP, d_arrInd, d_childrens);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

        sumTime += gpu_elapsed_time_ms;


        cudaMemcpy(h_childrens, d_childrens, countInd * N * sizeof(double), cudaMemcpyDeviceToHost);
        begin = std::chrono::steady_clock::now();

        fitness(count, countInd, h_childrens, h_arrInd);
        Selection(h_indexes, h_arrInd, countInd, countParents, h_bP, h_childrens);

        end = std::chrono::steady_clock::now();

        elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        sumTime += elapsed_ms.count();

        //std::cout << "Generation " << generation << ": " << h_arrInd[0] << std::endl;

        indBest = generation;

        if (h_arrInd[0] < min) {
            min = h_arrInd[0];
            indBest = generation;
        }

        if (val == h_arrInd[0]) sameIter++;

        else {
            val = h_arrInd[0];
            sameIter = 1;
        }

        if (sameIter >= maxConstIter) {
            std::cout << "Same " << maxConstIter << " iterations" << std::endl;
            break;
        }

    }


    std::cout << "Time: " << sumTime << std::endl;

    std::cout << "Error: " << min << std::endl << "generation: " << indBest << std::endl;

    double* temp = (double*)malloc(N * sizeof(double));
    for (int j = 0; j < N; j++) {
        std::cout << h_bP[j] << "*x^" << j;
        if (j + 1 < N) {
            std::cout << "+ ";
        }
    }
    std::cout << std::endl;

    cudaFree(d_arrInd);
    cudaFree(d_bP);
    cudaFree(d_childrens);

}
