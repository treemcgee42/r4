//
//  Compute.metal
//  r4
//
//  Created by Varun Malladi on 6/15/24.
//

#include <metal_stdlib>
using namespace metal;

struct Camera {
    float3 position;
    float3 direction;
    float fov;
};

kernel void
rayTraceSphere(texture2d<float, access::write> outTexture [[texture(0)]],
                 constant Camera *camera [[buffer(0)]],
                 uint2 gid [[thread_position_in_grid]]) {
    float width = outTexture.get_width();
    float height = outTexture.get_height();
    float aspectRatio = width / height;
    
    // --- Compute UV coordinates.
    // Assume we are in the center of the pixel. Normalize to [-1, 1].
    float2 uv = (float2(gid) + 0.5) / float2(width, height);
    uv = uv * 2.0 - 1.0;
    uv.x *= aspectRatio;
    
    float3 rayDir = normalize(float3(uv.x, uv.y, -1.0));
    float3 rayOrigin = camera->position;
    
    float3 sphereCenter = float3(0.0, 0.0, -5.0);
    float sphereRadius = 1.0;
    
    float3 oc = sphereCenter - rayOrigin;
    float a = dot(rayDir, rayDir);
    float b = -2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - 4 * a * c;
    
    float3 color = float3(0.0);
    if (discriminant >= 0) {
        float t = (-b - sqrt(discriminant)) / (2.0 * a);
        if (t > 0) {
            float3 hitPoint = rayOrigin + t * rayDir;
            float3 normal = normalize(hitPoint - sphereCenter);
            color = 0.5 * (normal + 1.0);
        }
    }
    
    outTexture.write(float4(color, 1.0), gid);
}
