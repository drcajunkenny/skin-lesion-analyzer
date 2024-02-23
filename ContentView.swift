import SwiftUI
import CoreML
import Vision

struct ContentView: View {
    @State private var image: Image?
    @State private var showingImagePicker = false
    @State private var inputImage: UIImage?
    @State private var usingCamera = false
    @State private var classificationResult: String = "No classification yet"
    
    var body: some View {
        NavigationView {
            VStack {
                Spacer()
                if image == nil {
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundColor(.orange)
                                        .font(.largeTitle)
                                        .padding()
                                    Text("This app is for educational purposes only and not for diagnostic use.")
                                        .foregroundColor(.orange)
                                        .padding()
                                }
                                
                                image?.resizable().scaledToFit()
                                
                                Spacer()
                                
                                Text(classificationResult).padding()
                Text("Take a picture of the lesion on your skin or select one from your photo library.")
                    .padding()
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                
                HStack {
                    Button(action: {
                        self.usingCamera = true
                        self.showingImagePicker = true
                    }) {
                        Image(systemName: "camera").font(.largeTitle).padding().background(Color.blue).foregroundColor(.white).clipShape(Circle())
                    }.padding().accessibility(label: Text("Take a photo"))
                    
                    Button(action: {
                        self.usingCamera = false
                        self.showingImagePicker = true
                    }) {
                        Image(systemName: "photo.on.rectangle.angled").font(.largeTitle).padding().background(Color.green).foregroundColor(.white).clipShape(Circle())
                    }.padding().accessibility(label: Text("Choose a photo"))
                }.padding()
            }.navigationBarTitle("Skin Lesion Analyzer", displayMode: .inline)
                .navigationBarTitleDisplayMode(.large)
                .sheet(isPresented: $showingImagePicker) {
                    ImagePicker(selectedImage: $inputImage, sourceType: usingCamera ? .camera : .photoLibrary)
                }
        }.onChange(of: inputImage) { _ in
            loadImage()
        }
    }
    
    func loadImage() {
        guard let inputImage = inputImage else { return }
        image = Image(uiImage: inputImage)
        classifyImage(inputImage)
    }
    
    func classifyImage(_ inputImage: UIImage) {
        guard let buffer = preprocessImage(inputImage: inputImage) else {
            classificationResult = "Error converting image."
            return
        }
        
        do {
            let model = try SkinImageClassifier(configuration: MLModelConfiguration())
            let predictionOutput = try model.prediction(image: buffer)
            let label = predictionOutput.target
            let probabilities = predictionOutput.targetProbability
            let probability = probabilities[label, default: 0]
            let formattedProbability = String(format: "%.2f%%", probability * 100)
            
            DispatchQueue.main.async {
                self.classificationResult = "\(label) with a confidence of \(formattedProbability)."
            }
        } catch {
            DispatchQueue.main.async {
                self.classificationResult = "Failed to make prediction: \(error.localizedDescription)"
            }
        }
    }
    
    func preprocessImage(inputImage: UIImage) -> CVPixelBuffer? {
        let targetSize = CGSize(width: 299, height: 299) // Update the target size based on the model's requirement
        
        guard let resizedImage = inputImage.resized(to: targetSize),
              let pixelBuffer = resizedImage.pixelBuffer() else {
            return nil
        }
        
        return pixelBuffer
    }
}

extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func pixelBuffer() -> CVPixelBuffer? {
        guard let image = self.cgImage else { return nil }
        
        let width = 224
        let height = 224
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width,
                                         height,
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)
        
        guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        
        let context = CIContext(options: nil)
        context.render(CIImage(cgImage: image), to: pixelBuffer)
        
        return pixelBuffer
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
