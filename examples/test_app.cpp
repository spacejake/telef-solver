#include "solver/gpu/gpuResidualFunction.h"

namespace {
    using namespace telef::solver;
}
/**
 * Continuously get frames of point cloud and image.
 *
 * Prints size of pointcloud and size of the image on every frame received
 * Remove all points that have NaN Position on Receive.
 * You can check this by watching varying number of pointcloud size
 */

int main(int ac, char* av[])
{
    std::vector<int> params = {1, 2};
    int nRes = 4;
    GPUResidualBlock testRB(nRes, params);
}

