//
//  VoxelVolumeSystem.swift
//  r4
//
//  Created by Varun Malladi on 6/16/24.
//

import Foundation
import Metal

struct BoundingBox {
    var min: MTLPackedFloat3 = MTLPackedFloat3()
    var max: MTLPackedFloat3 = MTLPackedFloat3()
    
    init(corner1: SIMD3<Float>, corner2: SIMD3<Float>) {
        self.min.x = corner1.x
        self.min.y = corner1.y
        self.min.z = corner1.z
        
        self.max.x = corner2.x
        self.max.y = corner2.y
        self.max.z = corner2.z
    }
}

struct BoundingBoxBuffer {
    var buffer: any MTLBuffer
    var count: Int
    var stride: Int
}

struct AccelerationStructure {
    var accelerationStructure: MTLAccelerationStructure
    var descriptor: MTLAccelerationStructureDescriptor
    var scratchBuffer: any MTLBuffer
}

struct ComputePipelineState {
    var computePipelineState: MTLComputePipelineState
    var functionTable: MTLIntersectionFunctionTable
}

typealias VoxelVolumeId = Int

class VoxelVolumeSystem {
    var boundingBoxes: [BoundingBox] = []
    
    func createEmptyVoxelVolume(corner1: SIMD3<Float>,
                                corner2: SIMD3<Float>) -> VoxelVolumeId {
        self.boundingBoxes.append(BoundingBox(corner1: corner1, corner2: corner2))
        return self.boundingBoxes.count
    }
    
    func createBoundingBoxBuffer(device: MTLDevice) -> BoundingBoxBuffer? {
        let bufferSize = self.boundingBoxes.count * MemoryLayout<BoundingBox>.stride
        guard let buffer = device.makeBuffer(bytes: self.boundingBoxes,
                                             length: bufferSize) else {
            return nil
        }
        return BoundingBoxBuffer(buffer: buffer, 
                                 count: self.boundingBoxes.count,
                                 stride: MemoryLayout<BoundingBox>.stride)
    }
    
    func createComputePipeline(device: MTLDevice, shaderLibrary: MTLLibrary) -> ComputePipelineState? {
        guard let boundingBoxIntersectionFunction =
                shaderLibrary.makeFunction(name: "boundingBoxIntersectionFunction"),
              let computeFunction = shaderLibrary.makeFunction(name: "rtKernel") else {
            return nil
        }
        let linkedFunctions = MTLLinkedFunctions()
        linkedFunctions.functions = [ boundingBoxIntersectionFunction ]
        
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = computeFunction
        descriptor.linkedFunctions = linkedFunctions
        
        var computePipeline: MTLComputePipelineState
        do {
            computePipeline = try device.makeComputePipelineState(
                descriptor: descriptor,
                options: [],
                reflection: nil)
        } catch {
            return nil
        }
        
        let intersectionFunctionTableDescriptor = MTLIntersectionFunctionTableDescriptor()
        intersectionFunctionTableDescriptor.functionCount = 1
        guard let functionTable = computePipeline.makeIntersectionFunctionTable(
            descriptor: intersectionFunctionTableDescriptor) else {
            return nil
        }
        functionTable.setFunction(
            computePipeline.functionHandle(function: boundingBoxIntersectionFunction),
            index: 0)
        
        return ComputePipelineState(
            computePipelineState: computePipeline,
            functionTable: functionTable)
    }
    
    func makeAccelerationStructure(device: MTLDevice) -> AccelerationStructure? {
        guard let boundingBoxBuffer = self.createBoundingBoxBuffer(device: device) else {
            return nil
        }
        
        let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
        
        let geometryDescriptor = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
        geometryDescriptor.boundingBoxBuffer = boundingBoxBuffer.buffer
        geometryDescriptor.boundingBoxCount = boundingBoxBuffer.count
        geometryDescriptor.intersectionFunctionTableOffset = 0
        geometryDescriptor.primitiveDataBuffer = boundingBoxBuffer.buffer
        geometryDescriptor.primitiveDataStride = boundingBoxBuffer.stride
        geometryDescriptor.primitiveDataElementSize = MemoryLayout<BoundingBox>.size
        descriptor.geometryDescriptors = [geometryDescriptor]
        
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard let accelerationStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratchBuffer = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate) else {
            return nil
        }

        return AccelerationStructure(accelerationStructure: accelerationStructure,
                                     descriptor: descriptor,
                                     scratchBuffer: scratchBuffer)
    }
}
