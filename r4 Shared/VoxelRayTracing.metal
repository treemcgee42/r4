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
    intersection = intersector.intersect(r, accelerationStructure, functionTable);
    
    float3 color = float3(0);
    if (intersection.type == intersection_type::bounding_box) {
        float x = float(threadPositionInGrid.x) / outTexture.get_width();
        float y = float(threadPositionInGrid.y) / outTexture.get_height();
        color = float3(x, y, 0);
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
};

#define EPSILON 0.000001
#define IS_ZERO(x) (fabs(x) < EPSILON)
#define IS_POSITIVE(x) ((x) > EPSILON)
#define IS_NEGATIVE(x) ((x) < -EPSILON)

struct VoxelVolumeIntersectionResult
amanatidesWooAlgorithm(float3 rayOrigin,
                       float3 rayDirection,
                       float rayIntersectionTMin,
                       float rayIntersectionTMax,
                       float3 voxelSize,
                       float3 boxMin,
                       float3 boxMax,
                       Grid3dView grid3dView,
                       const device int* grid3dData) {
    float tMin = rayIntersectionTMin;
    float tMax = rayIntersectionTMax;
    const float3 rayStart = rayOrigin + rayDirection * tMin;
    
    float3 currentIdxPrecise = (rayStart - boxMin) / voxelSize;
    float3 fractionalPart = currentIdxPrecise - floor(currentIdxPrecise);
    int3 currentIdx = int3(floor(currentIdxPrecise));
    if (IS_ZERO(fractionalPart.x - 1.f)) {
        currentIdx.x += 1;
    }
    if (IS_ZERO(fractionalPart.y - 1.f)) {
        currentIdx.y += 1;
    }
    if (IS_ZERO(fractionalPart.z - 1.f)) {
        currentIdx.z += 1;
    }

    // For the purposes of the algorithm, we consider a point to be inside a voxel
    // if, componentwise, voxelMinCorner <= p < voxelMaxCorner.
    
    int3 steps;
    steps.x = IS_POSITIVE(rayDirection.x) - IS_NEGATIVE(rayDirection.x);
    steps.y = IS_POSITIVE(rayDirection.y) - IS_NEGATIVE(rayDirection.y);
    steps.z = IS_POSITIVE(rayDirection.z) - IS_NEGATIVE(rayDirection.z);
    
    float3 tDeltas = abs(voxelSize / rayDirection);
    
    // (currentIdx + off) is, componentwise, the index of the next voxel
    // boundary in the direction of the ray.
    //
    // Consider voxelSize = (1, 1, 1) and we're at point (0, 1.5, 3) with
    // rayDirection = (1, -0.2, -0.3). Then the point lies in voxel
    // (0, 1, 3). In this case, off should be (1, 0, -1).
    
    // Attempt 2
    // int3 off = int3(steps == int3(1));
    int3 off;
    if (steps.x == 1) {
        off.x = 1;
    } else {
        if (IS_ZERO(currentIdxPrecise.x - (float)currentIdx.x)) {
            off.x = -1;
        } else {
            off.x = 0;
        }
    }
    if (steps.y == 1) {
        off.y = 1;
    } else {
        if (IS_ZERO(currentIdxPrecise.y - (float)currentIdx.y)) {
            off.y = -1;
        } else {
            off.y = 0;
        }
    }
    if (steps.z == 1) {
        off.z = 1;
    } else {
        if (IS_ZERO(currentIdxPrecise.z - (float)currentIdx.z)) {
            off.z = -1;
        } else {
            off.z = 0;
        }
    }
    const float3 nextBoundary = boxMin + float3(currentIdx + off) * voxelSize;
    float3 tMaxs = tMin + abs((nextBoundary - rayStart) / rayDirection);
//    // Attempt 1
//    float3 tMaxs = tMin + abs(float3(currentIdx + max(int3(0), steps)) * voxelSize - (rayStart - boxMin)) / abs(rayDirection);
    
    if (IS_ZERO(rayDirection.x)) {
        steps.x = 0;
        tDeltas.x = tMax + 1;
        tMaxs.x = tMax + 1;
    }
    if (IS_ZERO(rayDirection.y)) {
        steps.y = 0;
        tDeltas.y = tMax + 1;
        tMaxs.y = tMax + 1;
    }
    if (IS_ZERO(rayDirection.z)) {
        steps.z = 0;
        tDeltas.z = tMax + 1;
        tMaxs.z = tMax + 1;
    }
    
    float distance = tMin;
    bool hit = false;
//    hit = hitVoxel(grid3dView, grid3dData,
//                   currentIdx.x, currentIdx.y, currentIdx.z);
//    if (hit) {
//        return { true, distance };
//    }
    while (currentIdx.x < (int)grid3dView.xExtent &&
           currentIdx.x >= 0 &&
           currentIdx.y < (int)grid3dView.yExtent &&
           currentIdx.y >= 0 &&
           currentIdx.z < (int)grid3dView.zExtent &&
           currentIdx.z >= 0) {
        hit = hitVoxel(grid3dView, grid3dData,
                       currentIdx.x, currentIdx.y, currentIdx.z);
        if (hit) {
            return { true, distance };
        }
        
        if (tMaxs.x < tMaxs.y) {
            if (tMaxs.x < tMaxs.z) {
                currentIdx.x += steps.x;
                distance = tMaxs.x;
                tMaxs.x += tDeltas.x;
            } else {
                currentIdx.z += steps.z;
                distance = tMaxs.z;
                tMaxs.z += tDeltas.z;
            }
        } else {
            if (tMaxs.y < tMaxs.z) {
                currentIdx.y += steps.y;
                distance = tMaxs.y;
                tMaxs.y += tDeltas.y;
            } else {
                currentIdx.z += steps.z;
                distance = tMaxs.z;
                tMaxs.z += tDeltas.z;
            }
        }
    }
    
    return { hit, distance };
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
                                                  const device int* grid3dData [[buffer(0)]]) {
    VoxelVolumeData voxelVolumeData = ((const device VoxelVolumeData*)perPrimitiveData)[primitiveIndex];
    
    BoundingBox box = voxelVolumeData.boundingBox;
    
    //  --- Ray-box intersection test
    
    const struct RayBoxIntersectionResult rayBoxIntersectionResult = rayBoxIntersection(origin, direction, minDistance, maxDistance, box.min, box.max);
    if (!rayBoxIntersectionResult.intersected) {
        return { false, 0.0 };
    }
    // return { true, rayBoxIntersectionResult.tEnterOrMin };
    
    // --- Voxel volume intersection

    const struct VoxelVolumeIntersectionResult voxelVolumeIntersectionResult = simpleRayMarch(origin, direction, rayBoxIntersectionResult.tEnterOrMin, rayBoxIntersectionResult.tExitOrMax, float3(1.0), box.min, box.max, voxelVolumeData.grid3dView, grid3dData);
    
    // ---
    
    
    return { voxelVolumeIntersectionResult.intersected, voxelVolumeIntersectionResult.distance };
}
