//
//  GameViewController.swift
//  r4 macOS
//
//  Created by Varun Malladi on 6/10/24.
//

import Cocoa
import MetalKit

// Our macOS specific view controller
class GameViewController: NSViewController {
    override var acceptsFirstResponder: Bool { return true }
    var mtkView: MTKView!
    var renderer: Renderer!
    
    override func keyDown(with event: NSEvent) {
        guard let characters = event.characters else { return }
        for character in characters {
            switch character {
            case "w":
                renderer.cameraPosition.z -= 1
            case "s":
                renderer.cameraPosition.z += 1
            case "a":
                renderer.cameraPosition.x -= 1
            case "d":
                renderer.cameraPosition.x += 1
            case " ":
                if event.modifierFlags.contains(.shift) {
                    renderer.cameraPosition.y -= 1
                } else {
                    renderer.cameraPosition.y += 1
                }
            default:
                super.keyDown(with: event)
                break
            }
        }
        //mtkView.setNeedsDisplay(mtkView.bounds)
    }
    
    override func viewDidAppear() {
        super.viewDidAppear()
        self.view.window?.makeFirstResponder(self)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View attached to GameViewController is not an MTKView")
            return
        }

        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }

        mtkView.device = defaultDevice

        guard let newRenderer = Renderer(metalKitView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)

        mtkView.delegate = renderer
    }
}
