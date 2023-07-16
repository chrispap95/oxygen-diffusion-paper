/*
Compile and run inside a Google Colab notebook with a GPU using the commands:

!nvidia-smi
!nvcc  -o diffusion -x cu -lnvToolsExt drive/MyDrive/path/to/file/diffusionCUDARevised.cu
!./diffusion

The first line is not strictly necesary, but it lets us check what GPU we have,
probably a Tesla T4.

The second line runs the compiller. It is recomended you put this file on your
Google Drive and mount your drive to the Colab session.

The third line runs the code. Every 100 steps it will print out the current step
number.
*/

#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <nvToolsExt.h>
#include "CLI/CLI.hpp"
#include <stdio.h>
#include <ctime>

// Use a struct for the configuration to make it easier to pass around
struct Config
{
    float diffCoeff;                  // Diffusion coefficient for oxygen in the polymer
    float radFormRate;                // Radical formation rate = the specific rate of radical formation times the dose rate (use only that for now)
    float k1;                         // Radical crosslinking rate
    float k2;                         // Radical oxidation rate
    float doseRate;                   // Dose rate
    int irrTime;                      // Irradiation time
    int dimT;                         // Number of time steps
    int dimX;                         // Size of x dimension of the array
    int dimY;                         // Size of y dimension of the array
    int dimZ;                         // Size of z dimension of the array
    int DSIZE;                        // Total size of the array
    int SSIZE;                        // Number of slices
    bool limitedSave;                 // Whether to save only the middle slice for activity
    std::string outputFileNamePrefix; // Prefix for the output file name
};

#define blocks 80    // Should be number of streaming multiprocessors x2
#define threads 1024 // Should probably be 1024

#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

// Function to print the configuration
void printConfig(const Config &config)
{
    std::cout << std::left << std::setw(26) << "Parameter"
              << "Value" << '\n';
    std::cout << std::string(40, '-') << '\n';
    std::cout << std::left << std::setw(26) << "Diffusion coefficient" << config.diffCoeff << '\n';
    std::cout << std::left << std::setw(26) << "Radical formation rate" << config.radFormRate << '\n';
    std::cout << std::left << std::setw(26) << "Radical crosslinking rate" << config.k1 << '\n';
    std::cout << std::left << std::setw(26) << "Radical oxidation rate" << config.k2 << '\n';
    std::cout << std::left << std::setw(26) << "Dose rate" << config.doseRate << '\n';
    std::cout << std::left << std::setw(26) << "Irradiation time" << config.irrTime << '\n';
    std::cout << std::left << std::setw(26) << "Number of time steps" << config.dimT << '\n';
    std::cout << std::left << std::setw(26) << "Dimensions"
              << "(" << config.dimX << ", " << config.dimY << ", " << config.dimZ << ")" << '\n';
    std::cout << std::left << std::setw(26) << "Total number of steps" << config.DSIZE << '\n';
    std::cout << std::left << std::setw(26) << "Number of slices" << config.SSIZE << '\n';
    std::cout << std::left << std::setw(26) << "Limited save" << config.limitedSave << '\n';
    std::cout << std::left << std::setw(26) << "Output file name prefix" << config.outputFileNamePrefix << '\n';
}

__global__ void finiteDiff(const float *inputVal, float *outputVal,
                           const float *inputRad, float *outputRad,
                           float *saveSlice, float *saveActivity,
                           Config config, int tStamp)
{

    for (int index = threadIdx.x + blockDim.x * blockIdx.x; index < config.DSIZE; index += gridDim.x * blockDim.x)
    {

        // Determine array location
        int x = index % config.dimX;
        int y = ((index - x) / config.dimX) % config.dimY;
        int z = (index - x - y * config.dimX) / config.dimY / config.dimX;
        float radicalLoss = 0;
        float crossLinking = 0;
        float irradiationOn = 1;
        if (tStamp > config.irrTime)
        {
            irradiationOn = 0;
        }

        // Assuming not a boundary of the array
        if (x > 0 && y > 0 && z > 0 && x < config.dimX - 1 && y < config.dimY - 1 && z < config.dimZ - 1)
        {
            // Oxygen concentration - initial condition
            outputVal[index] = inputVal[index];

            // Applying the laplacian for oxygen diffusion across the 3 dimensions
            outputVal[index] += config.diffCoeff * (inputVal[index - 1] + inputVal[index + 1] - 2 * inputVal[index]);
            outputVal[index] += config.diffCoeff * (inputVal[index - config.dimX] + inputVal[index + config.dimX] - 2 * inputVal[index]);
            outputVal[index] += config.diffCoeff * (inputVal[index - config.dimX * config.dimY] + inputVal[index + config.dimX * config.dimY] - 2 * inputVal[index]);

            // Radical concentration - initial condition + radical formation
            outputRad[index] = inputRad[index] + config.radFormRate * irradiationOn;

            // Radical oxidation calculation
            // Need to account for zero concentration cases
            // Assume that the radical loss consumes fully the lowest quantity
            radicalLoss = config.k2 * outputRad[index] * outputVal[index];
            if (radicalLoss > outputRad[index])
            {
                if (outputVal[index] > outputRad[index])
                {
                    radicalLoss = outputRad[index];
                }
                else
                {
                    radicalLoss = outputVal[index];
                }
            }
            else if (radicalLoss > outputVal[index])
            {
                radicalLoss = outputVal[index];
            }

            // Apply radical loss
            outputRad[index] -= radicalLoss;
            outputVal[index] -= radicalLoss;

            // Crosslinking
            crossLinking = config.k1 * outputRad[index] * outputRad[index];
            // Boundary conditions - this quantity cannot go below zero
            if (crossLinking > outputRad[index])
            {
                outputRad[index] = 0;
            }
            else
            {
                // Normal case
                outputRad[index] -= crossLinking;
            }
        }
        else
        {
            // Fixed concentration outside
            outputVal[index] = inputVal[index];
        }

        if (x == (int)config.dimX / 2)
        {
            if (config.limitedSave)
            {
                if (z == (int)config.dimZ / 2)
                {
                    saveSlice[tStamp * config.dimY + y] = outputVal[index];
                    saveActivity[tStamp * config.dimY + y] = radicalLoss;
                }
            }
            else
            {
                saveSlice[tStamp * config.dimY * config.dimZ + y + z * config.dimY] = outputVal[index];
                saveActivity[tStamp * config.dimY * config.dimZ + y + z * config.dimY] = radicalLoss;
            }
        }
    }
}

// Take one step of the algorithm
void stepAlgo(float *d_inputVal, float *d_outputVal, float *d_inputRad,
              float *d_outputRad, float *d_saveSlice, float *d_saveActivity, int tStamp, Config *config)
{

    // Run the main algorithm
    finiteDiff<<<blocks, threads>>>(d_inputVal, d_outputVal, d_inputRad,
                                    d_outputRad, d_saveSlice, d_saveActivity,
                                    *config, tStamp);
    cudaCheckErrors("main kernel launch failure");
    cudaDeviceSynchronize();
}

void run(float *inputVal, float *inputRad, float *saveSlice, float *saveActivity, Config *config)
{
    int counter = 1;

    // Declare device pointers
    float *d_inputVal;
    float *d_outputVal;
    float *d_inputRad;
    float *d_outputRad;
    float *d_saveSlice;
    float *d_saveActivity;

    // Allocate memory on the gpu
    cudaMalloc(&d_inputVal, config->DSIZE * sizeof(float));
    cudaMalloc(&d_outputVal, config->DSIZE * sizeof(float));
    cudaMalloc(&d_inputRad, config->DSIZE * sizeof(float));
    cudaMalloc(&d_outputRad, config->DSIZE * sizeof(float));
    cudaMalloc(&d_saveSlice, config->SSIZE * sizeof(float));
    cudaMalloc(&d_saveActivity, config->SSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); // error checking

    // Copy data to the GPU
    cudaMemcpy(d_inputVal, inputVal, config->DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputVal, inputVal, config->DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputRad, inputRad, config->DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputRad, inputRad, config->DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    std::time_t msStart = std::time(nullptr);
    while (counter <= config->dimT)
    {
        // Print out the counter every 1000 iterations
        if ((counter - 1) % 1000 == 0)
        {
            std::cout << counter - 1 << std::endl;
        }
        // Run 1 step of the algorithm
        stepAlgo(d_inputVal, d_outputVal, d_inputRad, d_outputRad, d_saveSlice, d_saveActivity, counter - 1, config);
        counter++;

        // Run another step but with the input and output arrays flipped so
        // the memory doesn't need copied
        stepAlgo(d_outputVal, d_inputVal, d_outputRad, d_inputRad, d_saveSlice, d_saveActivity, counter - 1, config);
        counter++;
    }
    std::time_t msEnd = std::time(nullptr);

    // Give timing information
    std::cout << double(msEnd - msStart) * double(1000) / double(counter) << " ms per step\n";

    // Copy data off the GPU
    cudaMemcpy(inputVal, d_inputVal, config->DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(saveSlice, d_saveSlice, config->SSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(saveActivity, d_saveActivity, config->SSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    // Free the memory on the GPU
    cudaFree(d_inputVal);
    cudaFree(d_outputVal);
    cudaFree(d_inputRad);
    cudaFree(d_outputRad);
    cudaFree(d_saveSlice);
    cudaFree(d_saveActivity);
}

int main(int argc, char **argv)
{
    CLI::App app{"Radical diffusion simulation"};
    Config config;

    // Set default values
    config.diffCoeff = 0.01;    // this is perhaps too high but it sppeds up the simulation
    config.radFormRate = 0.5;   // data show it be at least 30 times larger than the diffusion coefficient and up to 1000 times larger for higher dose rates
    config.k1 = 0.001;          // Leave this at this small value for now
    config.k2 = 1.5;            // according to data: k2^2 is at least 1000 times 4*k1*radFormRate
    config.doseRate = 1;        // not used for now
    config.irrTime = 10000;     // total time of irradiation. Is important for the total dose
    config.dimT = 20000;        // total time of the simulation. Make sure it is enough for full annealing
    config.limitedSave = false; // save everything by default
    std::vector<int> dimXYZ = {100, 100, 500};
    config.outputFileNamePrefix = "output";

    // Declare the command line options
    app.add_option("--diffCoeff", config.diffCoeff, "Diffusion coefficient");
    app.add_option("--radFormRate", config.radFormRate, "Radical formation rate");
    app.add_option("--k1", config.k1, "Rate of crosslinking");
    app.add_option("--k2", config.k2, "Rate of radical oxidation");
    app.add_option("--doseRate", config.doseRate, "Dose rate");
    app.add_option("--irrTime", config.irrTime, "Irradiation time");
    app.add_option("--totalTime", config.dimT, "Total time");
    app.add_option("--dimXYZ", dimXYZ, "Dimensions X Y Z of the array")->expected(3);
    app.add_option("--outputPrefix", config.outputFileNamePrefix, "Output file name");
    app.add_flag("-l,--limitedSave", config.limitedSave, "Save only the middle slice for activity");

    CLI11_PARSE(app, argc, argv);

    config.dimX = dimXYZ[0];
    config.dimY = dimXYZ[1];
    config.dimZ = dimXYZ[2];
    config.DSIZE = config.dimX * config.dimY * config.dimZ;
    config.SSIZE = config.dimY * config.dimZ * config.dimT;
    if (config.limitedSave)
    {
        config.SSIZE = config.dimY * config.dimT;
    }

    // Print out the config
    printConfig(config);

    // Allocate arrays for data storage
    float *inputArray = new float[config.DSIZE];
    float *radArray = new float[config.DSIZE];
    float *saveSlice = new float[config.SSIZE];
    float *saveActivity = new float[config.SSIZE];

    float inside = 1;  // Concentration inside
    float outside = 1; // Concentration outside
    float radVal = 0;  // Initial radical concentration

    // Initialize the first array
    for (int x = 0; x < config.dimX; x++)
    {
        for (int y = 0; y < config.dimY; y++)
        {
            for (int z = 0; z < config.dimZ; z++)
            {
                int index = x + config.dimX * (y + z * config.dimY); // Location in flat array
                if (x > 0 && y > 0 && z > 0 && x < config.dimX - 1 && y < config.dimY - 1 && z < config.dimZ - 1)
                {
                    // inside
                    inputArray[index] = inside;
                }
                else
                {
                    // outside
                    inputArray[index] = outside;
                }
                radArray[index] = radVal;
            }
        }
    }

    // Run the algorithm
    run(inputArray, radArray, saveSlice, saveActivity, &config);

    // Store the data in a binary file
    // This can be opened in python with:
    // np.fromfile("data.dat", dtype=np.float32)
    // data = np.reshape(data,(10000,100,100))
    // Shape is (t, y, x)
    if (!config.limitedSave)
    {
        std::string filenameOxygen = config.outputFileNamePrefix + "oxygen.dat";
        FILE *data = fopen(filenameOxygen.c_str(), "wb");
        fwrite(saveSlice, sizeof(float), config.SSIZE, data);
        fclose(data);
    }

    std::string filenameActivity = config.outputFileNamePrefix + "activity.dat";
    FILE *activity = fopen(filenameActivity.c_str(), "wb");
    fwrite(saveActivity, sizeof(float), config.SSIZE, activity);
    fclose(activity);

    // Free memory
    free(inputArray);
    free(radArray);
    free(saveSlice);
    free(saveActivity);

    return (0);
}
