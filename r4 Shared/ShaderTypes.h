//
//  ShaderTypes.h
//  r4 Shared
//
//  Created by Varun Malladi on 6/10/24.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexMeshPositions = 0,
    BufferIndexMeshGenerics  = 1,
    BufferIndexUniforms      = 2
};

typedef NS_ENUM(EnumBackingType, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
};

typedef NS_ENUM(EnumBackingType, TextureIndex)
{
    TextureIndexColor    = 0,
};

typedef struct {
    vector_float3 min;
    vector_float3 max;
} BoundingBox;

typedef struct {
    vector_float3 position;
    vector_float3 direction;
    float fov;
} Camera;

typedef struct {
    size_t startIdx;
    uint xExtent;
    uint yExtent;
    uint zExtent;
} Grid3dView;

typedef struct {
    Camera camera;
} Uniforms;

typedef struct {
    BoundingBox boundingBox;
    Grid3dView grid3dView;
} VoxelVolumeData;

#endif /* ShaderTypes_h */

