//
//  VoxelVolumeSystem.swift
//  r4
//
//  Created by Varun Malladi on 6/16/24.
//

import Foundation
import Metal

struct Buffer {
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

class Grid3dData {
    var data: [Int32] = []
    
    func createGrid3d(xExtent: Int,
                      yExtent: Int,
                      zExtent: Int,
                      initialValue: Int32) -> Grid3dView {
        let startIdx = self.data.count
        self.data.append(contentsOf: Array(repeating: initialValue,
                               count: xExtent * yExtent * zExtent))
        return Grid3dView(startIdx: size_t(startIdx),
                          xExtent: UInt32(xExtent),
                          yExtent: UInt32(yExtent),
                          zExtent: UInt32(zExtent))
    }

    func createGrid3d(min: SIMD3<Float>,
                      max: SIMD3<Float>,
                      initialValue: Int32) -> Grid3dView {
        let floors = floor(max - min)
        return self.createGrid3d(xExtent: Int(floors.x),
                                 yExtent: Int(floors.y),
                                 zExtent: Int(floors.z),
                                 initialValue: initialValue)
    }
    
    func getElement(view: Grid3dView, x: Int, y: Int, z: Int) -> Int32 {
        return self.data[view.index(x: x, y: y, z: z)]
    }
    
    func setElement(view: Grid3dView, x: Int, y: Int, z: Int, value: Int32) {
        self.data[view.index(x: x, y: y, z: z)] = value
    }
}

extension Grid3dView {
    func index(x: Int, y: Int, z: Int) -> Int {
        return Int(startIdx) + x + y * Int(xExtent) + z * Int(xExtent * yExtent);
    }

    func inBounds(x: Int, y: Int, z: Int) -> Bool {
        return x >= 0 && x < xExtent && y >= 0 && y < yExtent && z >= 0 && z < zExtent
    }
}

typealias VoxelVolumeId = Int

class VoxelVolumeSystem {
    var voxelVolumeDatas: [VoxelVolumeData] = []
    var grid3dData = Grid3dData()
    
    func createBufferGrid3dData(device: MTLDevice) -> Buffer? {
        let bufferSize = self.grid3dData.data.count * MemoryLayout<Int32>.stride
        guard let buffer = device.makeBuffer(bytes: self.grid3dData.data,
                                             length: bufferSize) else {
            return nil
        }
        return Buffer(buffer: buffer,
                      count: self.grid3dData.data.count,
                      stride: MemoryLayout<Int32>.stride)
    }
    
    func createBufferPrimitiveData(device: MTLDevice) -> Buffer? {
        let bufferSize = self.voxelVolumeDatas.count * MemoryLayout<VoxelVolumeData>.stride
        guard let buffer = device.makeBuffer(bytes: self.voxelVolumeDatas, 
                                             length: bufferSize) else {
            return nil
        }
        return Buffer(buffer: buffer,
                      count: self.voxelVolumeDatas.count,
                      stride: MemoryLayout<VoxelVolumeData>.stride)
    }
    
    func createComputePipeline(
        device: MTLDevice,
        shaderLibrary: MTLLibrary) -> ComputePipelineState? {
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
        
        let grids3dBuffer = self.createBufferGrid3dData(device: device)
        functionTable.setBuffer(grids3dBuffer?.buffer, offset: 0, index: 0)
        
        return ComputePipelineState(
            computePipelineState: computePipeline,
            functionTable: functionTable)
    }
    
    func createEmptyVoxelVolume(min: SIMD3<Float>,
                                max: SIMD3<Float>) -> VoxelVolumeId {
        let boundingBox = BoundingBox(min: min, max: max)
        let grid3dView = self.grid3dData.createGrid3d(
            min: min, max: max,
            initialValue: 0)
        self.voxelVolumeDatas.append(VoxelVolumeData(
            boundingBox: boundingBox, grid3dView: grid3dView))
        
        return self.voxelVolumeDatas.count - 1
    }
    
    func createSphere(center: SIMD3<Float>,
                      radius: Float) -> VoxelVolumeId {
        let volumeId = createEmptyVoxelVolume(min: center - SIMD3<Float>(repeating: radius),
                                              max: center + SIMD3<Float>(repeating: radius))
        let voxelData = self.voxelVolumeDatas[volumeId]
        let grid3dView = voxelData.grid3dView
        let boundingBox = voxelData.boundingBox
        
        for x in 0..<grid3dView.xExtent {
            for y in 0..<grid3dView.yExtent {
                for z in 0..<grid3dView.zExtent {
                    let voxelCenter = SIMD3<Float>(
                        boundingBox.min.x + (Float(x) + 0.5),
                        boundingBox.min.y + (Float(y) + 0.5),
                        boundingBox.min.z + (Float(z) + 0.5)
                    )
                    let distance = simd_length(voxelCenter - center)
                    let value = (distance <= radius) ? 1 : 0
                    self.grid3dData.setElement(view: grid3dView,
                                               x: Int(x), y: Int(y), z: Int(z),
                                               value: Int32(value))
                }
            }
        }
        
        return volumeId
    }
    
    func makeAccelerationStructure(device: MTLDevice) -> AccelerationStructure? {
        guard let primitiveDataBuffer = self.createBufferPrimitiveData(device: device) else {
            return nil
        }
        
        let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
        
        let geometryDescriptor = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
        // The first field of the struct is the bounding box.
        geometryDescriptor.boundingBoxBuffer = primitiveDataBuffer.buffer
        geometryDescriptor.boundingBoxCount = primitiveDataBuffer.count
        geometryDescriptor.boundingBoxStride = primitiveDataBuffer.stride
        geometryDescriptor.intersectionFunctionTableOffset = 0
        geometryDescriptor.primitiveDataBuffer = primitiveDataBuffer.buffer
        geometryDescriptor.primitiveDataStride = primitiveDataBuffer.stride
        geometryDescriptor.primitiveDataElementSize = MemoryLayout<VoxelVolumeData>.size
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
