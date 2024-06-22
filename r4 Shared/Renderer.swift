//
//  Renderer.swift
//  r4 Shared
//
//  Created by Varun Malladi on 6/10/24.
//

// Our platform independent renderer class

import Metal
import MetalKit
import simd

let maxBuffersInFlight = 3

class Renderer: NSObject, MTKViewDelegate {
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var voxelComputePipelineState: ComputePipelineState
    var computePipelineState: MTLComputePipelineState!
    var outputTexture: MTLTexture!
    var shaderLibrary: MTLLibrary
    
    
    var voxelVolumeSystem = VoxelVolumeSystem()
    var accelerationStructure: AccelerationStructure
    
    var uniformBuffer: MTLBuffer!
    var uniformBufferIdx = 0
    var uniformBufferOffset = 0
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    // SCENE
    var cameraPosition = SIMD3<Float>(0, 0, 1)
    var cameraDirection = SIMD3<Float>(0, 0, -1)
    var cameraFov: Float = 60 * .pi / 180
    
    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let size = metalKitView.drawableSize
        self.outputTexture = createTexture(device: self.device,
                                           width: Int(size.width),
                                           height: Int(size.height))
        
        // Necessary to support blitting.
        metalKitView.framebufferOnly = false

        guard let library = device.makeDefaultLibrary(),
              let kernel = library.makeFunction(name: "rayTraceSphere") else {
            print("Failed to create compute kernel")
            return nil
        }
        self.shaderLibrary = library
        
        do {
            self.computePipelineState = try device.makeComputePipelineState(function: kernel)
        } catch {
            print("Failed to create compute pipeline state: \(error)")
            return nil
        }
        
        // --- temporary for testing
        
//        _ = voxelVolumeSystem.createEmptyVoxelVolume(
//            min: SIMD3<Float>(-3, -3, -10),
//            max: SIMD3<Float>(2, 2, -5))
        _ = voxelVolumeSystem.createSphere(center: SIMD3<Float>(0, 0, -20), radius: 7)

        self.accelerationStructure = voxelVolumeSystem.makeAccelerationStructure(device: device)!
        self.voxelComputePipelineState = voxelVolumeSystem.createComputePipeline(device: device, shaderLibrary: library)!
        
        // --- end
        
        super.init()
        
        self.createUniformBuffers()
    }
    
    private func createUniformBuffers() {
        let size = MemoryLayout<Uniforms>.stride * maxBuffersInFlight;
        self.uniformBuffer = device.makeBuffer(length: size, options: .storageModeShared)
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        guard let drawable = view.currentDrawable else {
            return
        }
        render(to: drawable)
        
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
        self.outputTexture = createTexture(device: self.device,
                                           width: Int(size.width),
                                           height: Int(size.height))
    }
    
    func render(to drawable: CAMetalDrawable) {
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        // --- Build acceleration structure
        
        guard let accelerationCommandEncoder = commandBuffer.makeAccelerationStructureCommandEncoder() else {
            return
        }
        
        accelerationCommandEncoder.build(
            accelerationStructure: self.accelerationStructure.accelerationStructure,
            descriptor: self.accelerationStructure.descriptor,
            scratchBuffer: self.accelerationStructure.scratchBuffer,
            scratchBufferOffset: 0)
        accelerationCommandEncoder.endEncoding()
        
        // --- Rest
        
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }
        
        self.updateUniforms()
        
        commandEncoder.setComputePipelineState(self.voxelComputePipelineState.computePipelineState)
        
        // Bind buffers
        commandEncoder.setBuffer(self.uniformBuffer, offset: self.uniformBufferOffset, index: 0)
        commandEncoder.setAccelerationStructure(
            self.accelerationStructure.accelerationStructure,
            bufferIndex: 1)
        commandEncoder.setIntersectionFunctionTable(
            self.voxelComputePipelineState.functionTable,
            bufferIndex: 2)
        
        // Bind textures
        commandEncoder.setTexture(self.outputTexture, index: 0)

        // Dispatch threads
        let gridSize = MTLSize(width: self.outputTexture.width,
                               height: outputTexture.height,
                               depth: 1)
        let threadGroupSize = MTLSize(width: 8, height: 8, depth: 1)
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        commandEncoder.endEncoding()
        
        // --- Copy texture to framebuffer for displaying.
        
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.copy(
                from: outputTexture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: outputTexture.width, 
                                    height: outputTexture.height,
                                    depth: 1),
                to: drawable.texture,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blitEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func updateUniforms() {
        self.uniformBufferOffset = MemoryLayout<Uniforms>.stride * self.uniformBufferIdx
        let uniforms = self.uniformBuffer.contents().advanced(by: self.uniformBufferOffset).assumingMemoryBound(to: Uniforms.self)
        
        uniforms.pointee.camera.position = self.cameraPosition
        uniforms.pointee.camera.direction = self.cameraDirection
        uniforms.pointee.camera.fov = self.cameraFov
        
        self.uniformBufferIdx = (self.uniformBufferIdx + 1) % maxBuffersInFlight
    }
}

func createTexture(device: MTLDevice, width: Int, height: Int) -> (any MTLTexture)? {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm,
        width: width,
        height: height,
        mipmapped: false)
    descriptor.usage = [.shaderWrite, .shaderRead]
    return device.makeTexture(descriptor: descriptor)
}
