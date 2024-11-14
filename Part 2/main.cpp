#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <codecvt>
#include <mpi.h>
// #include <sys/_types/_u_char.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <regex>
#include<cmath>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// Define constants
#define TERMINATION_THRESHOLD 0.99
using namespace std;
// Function to read the dataset and reshape it
void readDataset(const std::string& filename, std::vector<float>& data, int& xDim, int& yDim, int& zDim) {
    // Extract dimensions from the filename using regex
    std::regex pattern(R"(Isabel_(\d+)x(\d+)x(\d+)_float32\.raw)");
    std::smatch match;
    
    if (!std::regex_match(filename, match, pattern)) {
        throw std::runtime_error("Filename does not match expected format.");
    }

    try {
        xDim = std::stoi(match[1].str());
        yDim = std::stoi(match[2].str());
        zDim = std::stoi(match[3].str());
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("Error parsing dimensions from filename.");
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open dataset file.");
    }

    size_t totalSize = static_cast<size_t>(xDim) * yDim * zDim;
    data.resize(totalSize);

    file.read(reinterpret_cast<char*>(data.data()), totalSize * sizeof(float));
    if (!file) {
        throw std::runtime_error("Error reading data from file.");
    }

    file.close();
}
long long count1 =0;

float opacityTF(float a){
 float intensity[] = {-4926.59326171875, -1310.073486328125, 103.00382995605469, 2593.85205078125};
    float opacity[] = {1.0, 0.75641030073165894, 0.0, 0.053846154361963272};

    int n = sizeof(intensity) / sizeof(intensity[0]);

    // Handle edge cases where 'a' is outside the given range
    if (a <= intensity[0]) {
        count1++;
        return opacity[0];  // Return max opacity for the lowest intensity
    }
    if (a >= intensity[n-1]) {
        count1++;
        return opacity[n-1];  // Return min opacity for the highest intensity
    }

    // Linear interpolation
    for (int i = 0; i < n - 1; ++i) {
        if (a >= intensity[i] && a <= intensity[i+1]) {
            // Perform linear interpolation
            float t = (a - intensity[i]) / (intensity[i+1] - intensity[i]);
            return opacity[i] + t * (opacity[i+1] - opacity[i]);
        }
    }

    // Should never reach here
    return 0.0f;
}

std::vector<float> colorTF(float a) {
    // Data points: intensity and corresponding (R, G, B) colors (normalized)
     float intensity[] = {
        -4926.59, -4803.33, -4680.07, -4556.81, -4433.55, -4310.28, -4187.02, -4063.76,
        -3940.5, -3817.24, -3693.97, -3570.71, -3447.45, -3324.19, -3200.93, -3077.66,
        -2954.4, -2831.14, -2707.88, -2584.62, -2466.28, -2461.35, -2461.28, -2461.28,
        -2333.28, -2205.27, -2077.27, -1949.26, -1821.26, -1693.26, -1565.25, -1437.25,
        -1309.24, -1181.24, -1053.24, -925.233, -797.229, -669.226, -541.222, -413.218,
        -285.214, -157.21, -29.2065, 98.8726, 348.371, 597.868, 847.366, 1096.86, 1346.36,
        1595.86, 1845.36, 2094.86, 2344.35, 2469.1, 2593.85
    };

     float R[] = {
        0.301961, 0.396078, 0.494118, 0.588235, 0.662745, 0.741176, 0.788235, 0.862745,
        0.901961, 0.917647, 0.925490, 0.937255, 0.945098, 0.952941, 0.964706, 0.968627,
        0.972549, 0.980392, 0.980392, 0.988235, 0.984314, 0.988235, 0.952941, 0.952941,
        0.890196, 0.827451, 0.776471, 0.725490, 0.678431, 0.631373, 0.580392, 0.537255,
        0.498039, 0.462745, 0.431373, 0.403922, 0.372549, 0.345098, 0.317647, 0.286275,
        0.254902, 0.231373, 0.200000, 0.149020, 0.200000, 0.247059, 0.305882, 0.372549,
        0.443137, 0.517647, 0.600000, 0.686275, 0.760784, 0.807843, 0.890196
    };

    float G[] = {
        0.047059, 0.0392157, 0.054902, 0.113725, 0.168627, 0.227451, 0.290196, 0.380392,
        0.458824, 0.521569, 0.580392, 0.643137, 0.709804, 0.768627, 0.827451, 0.878431,
        0.917647, 0.949020, 0.972549, 0.988235, 0.988235, 0.988235, 0.952941, 0.952941,
        0.890196, 0.823529, 0.764706, 0.713726, 0.662745, 0.607843, 0.556863, 0.505882,
        0.458824, 0.419608, 0.388235, 0.356863, 0.321569, 0.294118, 0.262745, 0.231373,
        0.200000, 0.172549, 0.145098, 0.196078, 0.254902, 0.317647, 0.388235, 0.458824,
        0.533333, 0.615686, 0.698039, 0.784314, 0.858824, 0.901961, 0.956863
    };

    float B[] = {
        0.090196, 0.0588235, 0.0352941, 0.0235294, 0.0156863, 0.00392157, 0.000000,
        0.0117647, 0.027451, 0.0470588, 0.0784314, 0.121569, 0.184314, 0.247059,
        0.325490, 0.423529, 0.513726, 0.596078, 0.670588, 0.756863, 0.854902,
        0.858824, 0.894118, 0.894118, 0.807843, 0.737255, 0.678431, 0.627451,
        0.580392, 0.533333, 0.486275, 0.443137, 0.407843, 0.372549, 0.345098,
        0.317647, 0.294118, 0.266667, 0.239216, 0.211765, 0.184314, 0.164706,
        0.137255, 0.278431, 0.345098, 0.415686, 0.494118, 0.568627, 0.643137,
        0.725490, 0.800000, 0.870588, 0.929412, 0.960784, 0.984314
    };

    int n = sizeof(intensity) / sizeof(intensity[0]);

    // If a is out of bounds, return the first or last color
    if (a <= intensity[0]) return std::vector<float>({(R[0]), (G[0]), (B[0])});
    if (a >= intensity[n - 1]) return std::vector<float>({(R[n - 1]), (G[n - 1]), (B[n - 1])});

    // Find the appropriate interval for interpolation
    for (int i = 0; i < n - 1; ++i) {
        if (a >= intensity[i] && a <= intensity[i + 1]) {
            // Linear interpolation factor
            float t = (a - intensity[i]) / (intensity[i + 1] - intensity[i]);
            
            // Interpolate each RGB component
            float red = ((R[i] + t * (R[i + 1] - R[i])));
            float green = ((G[i] + t * (G[i + 1] - G[i])));
            float blue = ((B[i] + t * (B[i + 1] - B[i])));
            
            return std::vector<float>{red, green, blue};
        }
    }

    // Default return, though this shouldn't happen
    return std::vector<float>{0, 0, 0};
}
void saveImageFromVector(const std::vector<float>& data, int width, int height, int channels, const std::string& filename) {
    // Ensure the data is normalized to 0-255 range
    std::vector<u_char> imageData(data.size());
    float minVal = *std::min_element(data.begin(), data.end());
    float maxVal = *std::max_element(data.begin(), data.end());

    for (size_t i = 0; i < data.size(); ++i) {
        imageData[i] = static_cast<u_char>(255 * (data[i] - minVal) / (maxVal - minVal));
    }

    int stride = width * 3;
    if (!stbi_write_png(filename.c_str(), width, height, 3, imageData.data(), stride)) {
        std::cerr << "Error: Failed to save the image as PNG." << std::endl;
    } else {
        std::cout << "Image saved successfully as " << filename << std::endl;
    }
}


void rayCasting(int xStart, int xEnd, int yStart, int yEnd, int zStart, int zEnd,
                int zDim, int yDim, int xDim, const std::vector<float>& data, 
                float stepSize, std::vector<float>& result, int& earlyTerminatedRays,
                MPI_Comm zComm, int parentZProcess, int childZProcess) {
    int xSize = xEnd - xStart + 1;
    int ySize = yEnd - yStart + 1;
    result.resize(xSize * ySize * 3, 0.0f);
    earlyTerminatedRays = 0;
    
    #pragma omp parallel for reduction(+:earlyTerminatedRays)
    for (int x = xStart; x <= xEnd; ++x) {
        for (int y = yStart; y <= yEnd; ++y) {
            float accumulatedValue = 0.0f;  // Default initial opacity

            // If this is not the first Z process, receive initial opacity
            if (parentZProcess != MPI_PROC_NULL) {
                MPI_Recv(&accumulatedValue, 1, MPI_FLOAT, parentZProcess, 0, zComm, MPI_STATUS_IGNORE);
            }

            float r = 0, g = 0, b = 0;
            for (float z = zStart; z < zEnd; z += stepSize) {
                if (accumulatedValue >= TERMINATION_THRESHOLD) {
                    earlyTerminatedRays++;
                    break;
                }

                int zIdx = static_cast<int>(z);
                int zNext = zIdx + 1;
                if (zNext >= zEnd) break;

                float t = (z - zIdx) / (zNext - zIdx);
                int idx1 = x + y * xDim + zIdx * xDim * yDim;
                int idx2 = x + y * xDim + zNext * xDim * yDim;

                float interpolatedData = data[idx1] + t * (data[idx2] - data[idx1]);
                std::vector<float> rgb = colorTF(interpolatedData);
                float op = opacityTF(interpolatedData);

                if (z == zStart && accumulatedValue == 0.0f) {
                    r = rgb[0];
                    g = rgb[1];
                    b = rgb[2];
                    accumulatedValue = op;
                } else {
                    r = r + ((1 - accumulatedValue) * rgb[0] * op);
                    g = g + ((1 - accumulatedValue) * rgb[1] * op);
                    b = b + ((1 - accumulatedValue) * rgb[2] * op);
                    accumulatedValue = accumulatedValue + (1 - accumulatedValue) * op;
                }
            }

            // After updating accumulated opacity, send it to the child process
            if (childZProcess != MPI_PROC_NULL) {
                MPI_Send(&accumulatedValue, 1, MPI_FLOAT, childZProcess, 0, zComm);
            }

            int index = (x - xStart) + (y - yStart) * xSize;
            result[index * 3] = r;
            result[index * 3 + 1] = g;
            result[index * 3 + 2] = b;
        }
    }
}




void copySubResultToFinal(int xStart, int yStart, const std::vector<float>& subResult, 
                          std::vector<float>& finalResult, int xBoundMin,int xBoundMax, int yBoundMin,int yBoundMax, 
                          int xSize, int ySize, int zDim) {
    cout << "xStart" << xStart << "yStart" << yStart <<"xSize" << xSize << "ySize" << ySize << endl;
    cout << "xBoundMin" << xBoundMin << "yBoundMin" << yBoundMin << "yBoundMax" << yBoundMax << endl;
    for (int i = 0; i < xSize; ++i) {
        for (int j = 0; j < ySize; ++j) {
            int finalIndex = ((xStart - xBoundMin + i) + (yStart - yBoundMin + j)*(xBoundMax-xBoundMin+1)) * 3;
            int subIndex = ( i+ j*xSize) * 3;
            finalResult[finalIndex] = subResult[subIndex];
            finalResult[finalIndex + 1] = subResult[subIndex + 1];
            finalResult[finalIndex + 2] = subResult[subIndex + 2];
        }
    }
}


// Function to transpose the image data
std::vector<float> transposeImage(const std::vector<float>& input, int width, int height, int channels) {
    std::vector<float> transposed(width * height * channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                // Calculate the index in the original image
                int originalIndex = (y * width + x) * channels + c;
                // Calculate the index in the transposed image
                int transposedIndex = (x * height + y) * channels + c;
                transposed[transposedIndex] = input[originalIndex];
            }
        }
    }
    return transposed;
}
std::vector<float> flipAndTransposeImage(const std::vector<float>& input, int width, int height, int channels) {
    std::vector<float> flippedAndTransposed(width * height * channels);

    // Compute the maximum x and y bounds
    int maxX = width - 1;
    int maxY = height - 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                // Calculate the index in the original image
                int originalIndex = (y * width + x) * channels + c;
                // Calculate the new flipped coordinates
                int flippedX = maxX - x;
                int flippedY = maxY - y;
                // Calculate the index in the flipped image
                int flippedIndex = (flippedY * width + flippedX) * channels + c;
                flippedAndTransposed[flippedIndex] = input[originalIndex];
            }
        }
    }
    return flippedAndTransposed;
}

void compositeWithOpacity(std::vector<float>& front, const std::vector<float>& back) {
    #pragma omp simd
    for (size_t i = 0; i < front.size(); i += 3) {
        float alpha_front = front[i + 2];
        float alpha_back = back[i + 2];
        
        // Front-to-back compositing
        front[i] = front[i] + (1 - alpha_front) * back[i];
        front[i + 1] = front[i + 1] + (1 - alpha_front) * back[i + 1];
        front[i + 2] = alpha_front + (1 - alpha_front) * alpha_back;
    }
}

float getFinalOpacity(const std::vector<float>& subResult) {
    float totalOpacity = 0.0f;
    for (size_t i = 2; i < subResult.size(); i += 3) { // Assuming RGB format, opacity is the third component
        totalOpacity += subResult[i]; // Accumulate opacity values
    }
    return totalOpacity; // Return the total accumulated opacity
}

// Function to perform domain decomposition and ray casting
void volumeRendering(MPI_Comm comm, int xDim, int yDim, int zDim, const std::vector<float>& data, 
                    float stepSize, int xBoundMin, int xBoundMax, int yBoundMin, int yBoundMax,
                    int procX, int procY, int procZ) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int pz = rank / (procX * procY);
    int remaining = rank % (procX * procY);
    int py = remaining / procX;
    int px = remaining % procX;
    
    int xChunkSize = (xBoundMax - xBoundMin + 1) / procX;
    int yChunkSize = (yBoundMax - yBoundMin + 1) / procY;
    int zChunkSize = zDim / procZ;
    
    int xStart = xBoundMin + px * xChunkSize;
    int xEnd = (px == procX - 1) ? xBoundMax : xStart + xChunkSize - 1;
    
    int yStart = yBoundMin + py * yChunkSize;
    int yEnd = (py == procY - 1) ? yBoundMax : yStart + yChunkSize - 1;
    
    int zStart = pz * zChunkSize;
    int zEnd = (pz == procZ - 1) ? zDim - 1 : zStart + zChunkSize - 1;

    int xSize = xEnd - xStart + 1;
    int ySize = yEnd - yStart + 1;
    std::vector<float> subResult;
    int earlyTerminatedRays = 0;
    float initialOpacity = 0.0f;

    int parentZProcess = (pz == 0) ? MPI_PROC_NULL : rank - procX * procY;
    int childZProcess = (pz == procZ - 1) ? MPI_PROC_NULL : rank + procX * procY;

    rayCasting(xStart, xEnd, yStart, yEnd, zStart, zEnd, zDim, yDim, xDim, 
               data, stepSize, subResult, earlyTerminatedRays, comm, parentZProcess, childZProcess);

    auto end = std::chrono::high_resolution_clock::now();
    double localTime = std::chrono::duration<double>(end - start).count();

    // Calculate the fraction of early ray terminations
    double totalRays = (xEnd - xStart)*(yEnd - yStart); // Total number of rays processed in the raycasting code
    double earlyTerminationFraction = earlyTerminatedRays / totalRays; // Fraction calculation

    // Gather all execution times and early ray termination fractions
    std::vector<double> allTimes(procX * procY * procZ);
    std::vector<double> allTerminationFractions(procX * procY * procZ);
    MPI_Gather(&localTime, 1, MPI_DOUBLE, allTimes.data(), 1, MPI_DOUBLE, 0, comm);
    MPI_Gather(&earlyTerminationFraction, 1, MPI_DOUBLE, allTerminationFractions.data(), 1, MPI_DOUBLE, 0, comm);

    // if (pz > 0) {
    //     // For processes handling deeper Z sections, we need the accumulated opacity
    //     // from previous Z sections
    //     MPI_Recv(&initialOpacity, 1, MPI_FLOAT, (rank - procX * procY), 0, comm, MPI_STATUS_IGNORE);
    // }

    // // Modified rayCasting to use initial opacity
    // rayCasting(xStart, xEnd, yStart, yEnd, zStart, zEnd, zDim, yDim, xDim, 
    //            data, stepSize, subResult, earlyTerminatedRays, initialOpacity);

    // // Send accumulated opacity to next Z process
    // if (pz < procZ - 1) {
    //     float finalOpacity = getFinalOpacity(subResult);  // Get accumulated opacity
    //     MPI_Send(&finalOpacity, 1, MPI_FLOAT, (rank + procX * procY), 0, comm);
    // }

    // Z-direction compositing with proper opacity handling
    MPI_Comm z_comm;
    int z_color = rank % (procX * procY);
    MPI_Comm_split(comm, z_color, rank, &z_comm);

    int z_rank;
    MPI_Comm_rank(z_comm, &z_rank);

    // Modified compositing to respect front-to-back order
    if (z_rank == 0) { // Only the front-most Z processes participate in collecting results
        if (rank == 0) {
            // Initialize finalResult to store the complete image
            std::vector<float> finalResult((xBoundMax - xBoundMin + 1) * 
                                        (yBoundMax - yBoundMin + 1) * 3, 0.0f);
            
            // Copy the local subResult directly into the final result
            copySubResultToFinal(xStart, yStart, subResult, finalResult, 
                                xBoundMin, xBoundMax, yBoundMin, yBoundMax, 
                                xSize, ySize, zDim);

            // Receive and integrate results from other processes
            for (int i = 1; i < procX * procY; i++) {
                int recvXStart, recvXEnd, recvYStart, recvYEnd;
                MPI_Recv(&recvXStart, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
                MPI_Recv(&recvXEnd, 1, MPI_INT, i, 1, comm, MPI_STATUS_IGNORE);
                MPI_Recv(&recvYStart, 1, MPI_INT, i, 2, comm, MPI_STATUS_IGNORE);
                MPI_Recv(&recvYEnd, 1, MPI_INT, i, 3, comm, MPI_STATUS_IGNORE);

                int recvXSize = recvXEnd - recvXStart + 1;
                int recvYSize = recvYEnd - recvYStart + 1;
                std::vector<float> recvSubResult(recvXSize * recvYSize * 3); // Store the received result

                MPI_Recv(recvSubResult.data(), recvSubResult.size(), MPI_FLOAT, 
                        i, 4, comm, MPI_STATUS_IGNORE);

                copySubResultToFinal(recvXStart, recvYStart, recvSubResult, finalResult, 
                                    xBoundMin, xBoundMax, yBoundMin, yBoundMax,
                                    recvXSize, recvYSize, zDim);
            }

            // Save the final composited image
            std::string filename = std::to_string(procX) + "_" + 
                                std::to_string(procY) + "_" + 
                                std::to_string(procZ) + ".png";
            saveImageFromVector(finalResult, xBoundMax - xBoundMin + 1,
                                yBoundMax - yBoundMin + 1, 3, filename);

            // Write additional data to the text file
            std::ofstream outFile(std::to_string(procX) + "_" + 
                                std::to_string(procY) + "_" + 
                                std::to_string(procZ) + "_"+to_string(stepSize)+ ".txt", std::ios::app);
            if (outFile.is_open()) {
                outFile << "Total execution times for processor " << rank <<" (in seconds):\n";
                for (double time : allTimes) {
                    outFile << time << " ";
                }
                outFile << "\n";

                outFile << "Fraction of early ray termination for processor " << rank << ":\n";
                for (double fraction : allTerminationFractions) {
                    outFile << fraction << " ";
                }
                outFile << "\n";

                outFile.close();
            }
        } else {
            // Send the subResult and region bounds to process 0
            MPI_Send(&xStart, 1, MPI_INT, 0, 0, comm);
            MPI_Send(&xEnd, 1, MPI_INT, 0, 1, comm);
            MPI_Send(&yStart, 1, MPI_INT, 0, 2, comm);
            MPI_Send(&yEnd, 1, MPI_INT, 0, 3, comm);
            MPI_Send(subResult.data(), subResult.size(), MPI_FLOAT, 0, 4, comm);
        }
    }

    // Measure and output maximum execution time across all processes
    double maxTime;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        std::cout << "Total execution time: " << maxTime << " seconds" << std::endl;
    }

    // Clean up the Z communicator
    MPI_Comm_free(&z_comm);

}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np <num_processes> ./executable "
                     << "<dataset> <procX> <procY> <procZ> <stepSize>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // Parse input arguments
    std::string datasetFile = argv[1];
    int procX = std::stoi(argv[2]);
    int procY = std::stoi(argv[3]);
    int procZ = std::stoi(argv[4]);
    float stepSize = std::stof(argv[5]);

    // Variables to store dataset dimensions
    int xDim, yDim, zDim;
    std::vector<float> data;

    if (size != procX * procY * procZ) {
        if (rank == 0) {
            std::cerr << "Error: Number of processes must equal PX * PY * PZ" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        // Read the dataset on rank 0
        readDataset(datasetFile, data, xDim, yDim, zDim);
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&xDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zDim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Non-root processes allocate memory for data
    if (rank != 0) {
        data.resize(xDim * yDim * zDim);
    }

    // Broadcast the actual data
    MPI_Bcast(data.data(), xDim * yDim * zDim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Calculate bounds for the entire volume
    int xBoundMin = 0;
    int xBoundMax = xDim - 1;
    int yBoundMin = 0;
    int yBoundMax = yDim - 1;

    if (rank == 0) {
        std::cout << "Dataset dimensions: " << xDim << "x" << yDim << "x" << zDim << std::endl;
        std::cout << "Process grid: " << procX << "x" << procY << "x" << procZ << std::endl;
        std::cout << "Step size: " << stepSize << std::endl;
    }

    // Perform volume rendering
    volumeRendering(MPI_COMM_WORLD, xDim, yDim, zDim, data, stepSize, 
                   xBoundMin, xBoundMax, yBoundMin, yBoundMax, 
                   procX, procY, procZ);

    MPI_Finalize();
    return 0;
}
