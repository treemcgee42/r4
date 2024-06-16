//
//  VoxelRayTracing.metal
//  r4
//
//  Created by Varun Malladi on 6/16/24.
//

#include <metal_stdlib>
using namespace metal;
using namespace metal::raytracing;

struct BoundingBoxResult {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

struct Camera {
    float3 position;
    float3 direction;
    float fov;
};

ray generateCameraRay(constant Camera *camera,
                      float outTextureWidth, float outTextureHeight,
                      uint2 threadPositionInGrid) {
    float aspectRatio = outTextureWidth / outTextureHeight;
    
    // --- Compute UV coordinates.
    // Assume we are in the center of the pixel. Normalize to [-1, 1].
    float2 uv = ((float2(threadPositionInGrid) + 0.5) /
                 float2(outTextureWidth, outTextureHeight));
    uv = uv * 2.0 - 1.0;
    uv.x *= aspectRatio;
    
    float3 rayDir = normalize(float3(uv.x, uv.y, -1.0));
    float3 rayOrigin = camera->position;
    
    return ray(rayOrigin, rayDir);
}

[[kernel]]
void rtKernel(primitive_acceleration_structure accelerationStructure [[buffer(0)]],
              intersection_function_table<> functionTable [[buffer(1)]],
              constant Camera *camera [[buffer(2)]],
              texture2d<float, access::write> outTexture [[texture(0)]],
              uint2 threadPositionInGrid [[thread_position_in_grid]]) {
    ray r = generateCameraRay(camera,
                              outTexture.get_width(), outTexture.get_height(),
                              threadPositionInGrid);
    
    intersector<> intersector;
    intersection_result<> intersection;
    intersection = intersector.intersect(r, accelerationStructure, functionTable);
    
    float3 color = float3(0);
    if (intersection.type == intersection_type::bounding_box) {
        if (intersection.distance > 4.f) {
            float x = float(threadPositionInGrid.x) / outTexture.get_width();
            float y = float(threadPositionInGrid.y) / outTexture.get_height();
            color = float3(x, y, 0);
        }
    }
    
    outTexture.write(float4(color, 1.0), threadPositionInGrid);
}

struct BoundingBox {
    MTLPackedFloat3 min;
    MTLPackedFloat3 max;
};

[[intersection(bounding_box)]]
BoundingBoxResult boundingBoxIntersectionFunction(float3 origin [[origin]],
                                                  float3 direction [[direction]],
                                                  float minDistance [[min_distance]],
                                                  float maxDistance [[max_distance]],
                                                  uint primitiveIndex [[primitive_id]],
                                                  const device void* perPrimitiveData [[primitive_data]]) {
    BoundingBox box;
    box = *(const device BoundingBox*)perPrimitiveData;
    
    //  --- Ray-box intersection test
    
    float3 invDir = 1.0 / direction;
    float3 tMin = (box.min - origin) * invDir;
    float3 tMax = (box.max - origin) * invDir;

    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);

    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);

    if (tNear > tFar || tFar < 0.0) {
        return { false, 0.0 };
    }

    if (tNear > maxDistance || tFar < minDistance) {
        return { false, 0.0 };
    }
    
    // ---

    float hitDistance = tNear > minDistance ? tNear : minDistance;
    return { true, hitDistance };
}
