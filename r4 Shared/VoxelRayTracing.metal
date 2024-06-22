//
//  VoxelRayTracing.metal
//  r4
//
//  Created by Varun Malladi on 6/16/24.
//

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;
using namespace metal::raytracing;

struct BoundingBoxResult {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

ray generateCameraRay(constant Camera& camera,
                      float outTextureWidth, float outTextureHeight,
                      uint2 threadPositionInGrid) {
    float aspectRatio = outTextureWidth / outTextureHeight;
    
    // --- Compute UV coordinates.
    // Assume we are in the center of the pixel. Normalize to [-1, 1].
    float2 uv = ((float2(threadPositionInGrid) + 0.5) /
                 float2(outTextureWidth, outTextureHeight));
    uv = uv * 2.0 - 1.0;
    uv.x *= aspectRatio;
    uv.y *= -1.0;
    
    float3 rayDir = normalize(float3(uv.x, uv.y, -1.0));
    float3 rayOrigin = camera.position;
    
    return ray(rayOrigin, rayDir);
}

[[kernel]]
void rtKernel(constant Uniforms& uniforms [[buffer(0)]],
              primitive_acceleration_structure accelerationStructure [[buffer(1)]],
              intersection_function_table<> functionTable [[buffer(2)]],
              texture2d<float, access::write> outTexture [[texture(0)]],
              uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    ray r = generateCameraRay(uniforms.camera,
                              outTexture.get_width(), outTexture.get_height(),
                              threadPositionInGrid);
    
    intersector<> intersector;
    intersection_result<> intersection;
    float3 normal;
    intersection = intersector.intersect(r, accelerationStructure, functionTable, normal);
    
    float3 color;
    if (intersection.type == intersection_type::bounding_box) {
        //color = float3(1, 0, 0);
        color = 0.5 * (normal + float3(1.0));
    } else {
        float a = 0.5 * (normalize(r.direction).y + 1.0);
        color = (1.0 - a) * float3(1.0) + a * float3(0.5, 0.7, 1.0);
    }
    
    outTexture.write(float4(color, 1.0), threadPositionInGrid);
}

struct RayBoxIntersectionResult {
    bool intersected;
    float tEnterOrMin;
    float tExitOrMax;
};

struct RayBoxIntersectionResult
rayBoxIntersection(float3 rayOrigin,
                   float3 rayDirection,
                   float minDistance,
                   float maxDistance,
                   float3 boxMin,
                   float3 boxMax) {
    struct RayBoxIntersectionResult toReturn;
    toReturn.intersected = false;
    
    const float xInvDir = 1.0 / rayDirection.x;
    if (xInvDir > 0.0) {
        toReturn.tEnterOrMin = (boxMin.x - rayOrigin.x) * xInvDir;
        toReturn.tExitOrMax = (boxMax.x - rayOrigin.x) * xInvDir;
    } else {
        toReturn.tEnterOrMin = (boxMax.x - rayOrigin.x) * xInvDir;
        toReturn.tExitOrMax = (boxMin.x - rayOrigin.x) * xInvDir;
    }
    
    float tYEnterOrMin, tYExitOrMax;
    const float yInvDir = 1.0 / rayDirection.y;
    if (yInvDir > 0.0) {
        tYEnterOrMin = (boxMin.y - rayOrigin.y) * yInvDir;
        tYExitOrMax = (boxMax.y - rayOrigin.y) * yInvDir;
    } else {
        tYEnterOrMin = (boxMax.y - rayOrigin.y) * yInvDir;
        tYExitOrMax = (boxMin.y - rayOrigin.y) * yInvDir;
    }
    
    if (toReturn.tEnterOrMin > tYExitOrMax ||
        tYEnterOrMin > toReturn.tExitOrMax) {
        return toReturn;
    }
    if (tYEnterOrMin > toReturn.tEnterOrMin) {
        toReturn.tEnterOrMin = tYEnterOrMin;
    }
    if (tYExitOrMax < toReturn.tExitOrMax) {
        toReturn.tExitOrMax = tYExitOrMax;
    }
    
    float tZEnterOrMin, tZExitOrMax;
    const float zInvDir = 1.0 / rayDirection.z;
    if (zInvDir > 0.0) {
        tZEnterOrMin = (boxMin.z - rayOrigin.z) * zInvDir;
        tZExitOrMax = (boxMax.z - rayOrigin.z) * zInvDir;
    } else {
        tZEnterOrMin = (boxMax.z - rayOrigin.z) * zInvDir;
        tZExitOrMax = (boxMin.z - rayOrigin.z) * zInvDir;
    }
    
    if (toReturn.tEnterOrMin > tZExitOrMax ||
        tZEnterOrMin > toReturn.tExitOrMax) {
        return toReturn;
    }
    if (tZEnterOrMin > toReturn.tEnterOrMin) {
        toReturn.tEnterOrMin = tZEnterOrMin;
    }
    if (tZExitOrMax < toReturn.tExitOrMax) {
        toReturn.tExitOrMax = tZExitOrMax;
    }
    
    toReturn.intersected = (toReturn.tEnterOrMin < maxDistance &&
                            toReturn.tExitOrMax > minDistance);
    return toReturn;
}

bool
hitVoxel(Grid3dView grid3dView, const device int* grid3dData,
         int x, int y, int z) {
    return grid3dData[grid3dView.startIdx + x + y * grid3dView.xExtent + z * grid3dView.xExtent * grid3dView.yExtent] > 0;
}

struct VoxelVolumeIntersectionResult {
    bool intersected;
    float distance;
    float3 normal;
};

#define EPSILON 0.000001
#define IS_ZERO(x) (fabs(x) < EPSILON)

// Parameters:
// - `rayIntersectionTMin`: the time at which the ray (described by
// `rayOrigin + t * rayDirectionNormalized`) intersects the voxel volume.
// - `rayIntersectionTMin`: the time at which the ray (described by
// `rayOrigin + t * rayDirectionNormalized`) exits the voxel volume.
// - `boxMin`: the coordinates of the minimal corner of the voxel volume,
//             in world space.
// - `boxMax`: the coordinates of the maximal corner of the voxel volume,
//             in world space.
struct VoxelVolumeIntersectionResult
amanatidesWooAlgorithm(float3 rayOrigin,
                       float3 rayDirectionNormalized,
                       float rayIntersectionTMin,
                       float rayIntersectionTMax,
                       float3 voxelSize,
                       float3 boxMin,
                       float3 boxMax,
                       Grid3dView grid3dView,
                       const device int* grid3dData) {
    const float tMin = rayIntersectionTMin;
    const float tMax = rayIntersectionTMax;
    const float3 rayStart = rayOrigin + rayDirectionNormalized * tMin;
    
    // For the purposes of the algorithm, we consider a point to be inside a voxel
    // if, componentwise, voxelMinCorner <= p < voxelMaxCorner.
    
    const float3 rayStartInVoxelSpace = rayStart - boxMin;
    // In voxel units, in voxel space. Clamp to account for precision errors.
    int3 currentIdx = clamp(int3(floor(rayStartInVoxelSpace / voxelSize)),
                            int3(0),
                            int3(grid3dView.xExtent - 1,
                                 grid3dView.yExtent - 1,
                                 grid3dView.zExtent - 1));

    int3 steps = int3(sign(rayDirectionNormalized));
    float3 tDeltas = abs(voxelSize / rayDirectionNormalized);
    
    // tMaxs is, componentwise, the (total) time it will take for the ray to enter
    // the next voxel.
    // To compute tMax for a component:
    // - If rayDirection is positive, then the next boundary is in the next voxel.
    // - If rayDirection is negative, the the next boundary is at the start of the
    //   same voxel.
    
    // Multiply by voxelSize to get back to units of t.
    float3 nextBoundaryInVoxelSpace = float3(currentIdx + int3(steps > int3(0))) * voxelSize;
    float3 tMaxs = tMin + (nextBoundaryInVoxelSpace - rayStartInVoxelSpace) / rayDirectionNormalized;
    
    if (IS_ZERO(rayDirectionNormalized.x)) {
        steps.x = 0;
        tDeltas.x = tMax + 1;
        tMaxs.x = tMax + 1;
    }
    if (IS_ZERO(rayDirectionNormalized.y)) {
        steps.y = 0;
        tDeltas.y = tMax + 1;
        tMaxs.y = tMax + 1;
    }
    if (IS_ZERO(rayDirectionNormalized.z)) {
        steps.z = 0;
        tDeltas.z = tMax + 1;
        tMaxs.z = tMax + 1;
    }
    
    float distance = tMin;
    float3 normal;
    float3 hitPoint = rayOrigin + rayDirectionNormalized * tMin;
    if (fabs(hitPoint.x - boxMin.x) < EPSILON) {
        normal = float3(-1, 0, 0);
    } else if (fabs(hitPoint.x - boxMax.x) < EPSILON) {
        normal = float3(1, 0, 0);
    } else if (fabs(hitPoint.y - boxMin.y) < EPSILON) {
        normal = float3(0, -1, 0);
    } else if (fabs(hitPoint.y - boxMax.y) < EPSILON) {
        normal = float3(0, 1, 0);
    } else if (fabs(hitPoint.z - boxMin.z) < EPSILON) {
        normal = float3(0, 0, -1);
    } else if (fabs(hitPoint.z - boxMax.z) < EPSILON) {
        normal = float3(0, 0, 1);
    }
    while (currentIdx.x < (int)grid3dView.xExtent &&
           currentIdx.x >= 0 &&
           currentIdx.y < (int)grid3dView.yExtent &&
           currentIdx.y >= 0 &&
           currentIdx.z < (int)grid3dView.zExtent &&
           currentIdx.z >= 0) {
        if (hitVoxel(grid3dView, grid3dData,
                     currentIdx.x, currentIdx.y, currentIdx.z)) {
            return { true, distance, normalize(normal) };
        }
        
        if (tMaxs.x < tMaxs.y) {
            if (tMaxs.x < tMaxs.z) {
                currentIdx.x += steps.x;
                distance = tMaxs.x;
                tMaxs.x += tDeltas.x;
                normal = float3(-steps.x, 0, 0);
            } else {
                currentIdx.z += steps.z;
                distance = tMaxs.z;
                tMaxs.z += tDeltas.z;
                normal = float3(0, 0, -steps.z);
            }
        } else {
            if (tMaxs.y < tMaxs.z) {
                currentIdx.y += steps.y;
                distance = tMaxs.y;
                tMaxs.y += tDeltas.y;
                normal = float3(0, -steps.y, 0);
            } else {
                currentIdx.z += steps.z;
                distance = tMaxs.z;
                tMaxs.z += tDeltas.z;
                normal = float3(0, 0, -steps.z);
            }
        }
    }
    
    return { false, 0.0, float3(0.0) };
}

struct VoxelVolumeIntersectionResult
simpleRayMarch(float3 rayOrigin,
               float3 rayDirection,
               float rayIntersectionTMin,
               float rayIntersectionTMax,
               float3 voxelSize,
               float3 boxMin,
               float3 boxMax,
               Grid3dView grid3dView,
               const device int* grid3dData) {
    // position ray enters voxel volume
    float3 rayPosition = rayOrigin + rayIntersectionTMin * rayDirection;
    float3 step = 0.2 * normalize(rayDirection) * voxelSize;  // Adjust step size to the voxel size
    float totalDistance = rayIntersectionTMax - rayIntersectionTMin;

    // Calculate number of steps to perform based on total distance and step length
    int steps = int(ceil(totalDistance / length(step)));

    for (int i = 0; i < steps; ++i) {
        int3 voxelIndex = int3((rayPosition - boxMin) / voxelSize);
        if (hitVoxel(grid3dView, grid3dData, voxelIndex.x, voxelIndex.y, voxelIndex.z)) {
            float distance = length(rayPosition - rayOrigin);
            return { true, distance };
        }
        rayPosition += step;
        
        if (rayPosition.x < boxMin.x || rayPosition.x >= boxMax.x ||
            rayPosition.y < boxMin.y || rayPosition.y >= boxMax.y ||
            rayPosition.z < boxMin.z || rayPosition.z >= boxMax.z) {
            break;
        }
    }

    return { false, 0.0f };
}

[[intersection(bounding_box)]]
BoundingBoxResult boundingBoxIntersectionFunction(float3 origin [[origin]],
                                                  float3 direction [[direction]],
                                                  float minDistance [[min_distance]],
                                                  float maxDistance [[max_distance]],
                                                  uint primitiveIndex [[primitive_id]],
                                                  const device void* perPrimitiveData [[primitive_data]],
                                                  const device int* grid3dData [[buffer(0)]],
                                                  ray_data float3 & normal [[payload]]) {
    VoxelVolumeData voxelVolumeData = ((const device VoxelVolumeData*)perPrimitiveData)[primitiveIndex];
    
    BoundingBox box = voxelVolumeData.boundingBox;
    
    //  --- Ray-box intersection test
    
    const struct RayBoxIntersectionResult rayBoxIntersectionResult = rayBoxIntersection(origin, normalize(direction), minDistance, maxDistance, box.min, box.max);
    if (!rayBoxIntersectionResult.intersected) {
        return { false, 0.0 };
    }
    // return { true, rayBoxIntersectionResult.tEnterOrMin };
    
    // --- Voxel volume intersection

    const struct VoxelVolumeIntersectionResult voxelVolumeIntersectionResult = amanatidesWooAlgorithm(origin, normalize(direction), rayBoxIntersectionResult.tEnterOrMin, rayBoxIntersectionResult.tExitOrMax, float3(1.0), box.min, box.max, voxelVolumeData.grid3dView, grid3dData);
    normal = voxelVolumeIntersectionResult.normal;
    
    // ---
    
    
    return { voxelVolumeIntersectionResult.intersected, voxelVolumeIntersectionResult.distance };
}
