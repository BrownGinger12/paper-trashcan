import cv2
import numpy as np
import ncnn
import os
import time

# ===========================
# CONFIGURATION
# ===========================
MODEL_PARAM = "model.ncnn.param"
MODEL_BIN = "model.ncnn.bin"
TEST_IMAGE = "test.jpg"  # Path to your test image with paper
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
TARGET_SIZE = 320

# ===========================
# YOLO DETECTOR
# ===========================
class YOLOv5Detector:
    def __init__(self, param_path, bin_path, target_size=320, conf_threshold=0.5, nms_threshold=0.45):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        
        print(f"Loading param from: {param_path}")
        param_result = self.net.load_param(param_path)
        print(f"Loading bin from: {bin_path}")
        bin_result = self.net.load_model(bin_path)
        
        if param_result != 0:
            print("❌ Failed to load .param file!")
            return
        if bin_result != 0:
            print("❌ Failed to load .bin file!")
            return
        
        print("✅ Model loaded successfully!")
        
        self.target_size = target_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = ['paper']
    
    def detect(self, img):
        img_h, img_w = img.shape[:2]
        
        # Prepare input
        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            img_w, img_h, self.target_size, self.target_size
        )
        
        # Normalize
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize([], norm_vals)
        
        # Inference
        ex = self.net.create_extractor()
        
        # Try common input layer names
        input_success = False
        for input_name in ["in0", "images", "data", "input"]:
            result = ex.input(input_name, mat_in)
            if result == 0:
                # print(f"✅ Using input layer: '{input_name}'")  # Commented to reduce spam
                input_success = True
                break
        
        if not input_success:
            print("❌ Failed to find input layer!")
            return []
        
        mat_out = ncnn.Mat()
        
        # Try common output layer names
        output_success = False
        for output_name in ["out0", "output0", "output", "451"]:
            result = ex.extract(output_name, mat_out)
            if result == 0:
                # print(f"✅ Using output layer: '{output_name}'")  # Commented to reduce spam
                # print(f"   Output shape: c={mat_out.c}, h={mat_out.h}, w={mat_out.w}")
                output_success = True
                break
        
        if not output_success:
            print("❌ Failed to find output layer!")
            return []
        
        # Post-process
        detections = self.post_process(mat_out, img_w, img_h)
        return detections
    
    def post_process(self, mat_out, img_w, img_h):
        try:
            c = mat_out.c
            h = mat_out.h
            w = mat_out.w
            
            # Convert to numpy
            if c == 1 and h > w:
                out = np.array(mat_out).reshape(h, w).T
            elif c == 1:
                out = np.array(mat_out).reshape(h, w)
            elif h == 1:
                out = np.array(mat_out).reshape(c, w).T
            else:
                out = np.array(mat_out).reshape(h, w)
            
            # print(f"Numpy output shape: {out.shape}")  # Commented to reduce spam
            
        except Exception as e:
            print(f"❌ Numpy conversion error: {e}")
            return []
        
        boxes = []
        confidences = []
        
        # print(f"\nScanning {len(out)} detection candidates...")  # Commented to reduce spam
        high_conf_count = 0
        
        for i, detection in enumerate(out):
            if len(detection) < 5:
                continue
            
            objectness = detection[4]
            
            if objectness > self.conf_threshold:
                high_conf_count += 1
                
                x_center = detection[0]
                y_center = detection[1]
                width = detection[2]
                height = detection[3]
                
                # Check if normalized (0-1) or absolute pixels
                if x_center <= 1.0 and y_center <= 1.0:
                    x_center *= img_w
                    y_center *= img_h
                    width *= img_w
                    height *= img_h
                
                x = int(x_center - width / 2)
                y = int(y_center - height / 2)
                
                if width > 0 and height > 0 and x >= 0 and y >= 0:
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(objectness))
        
        # print(f"Found {high_conf_count} detections above threshold {self.conf_threshold}")  # Commented
        
        if len(boxes) == 0:
            # print("❌ No valid detections")  # Commented
            return []
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            # print(f"✅ After NMS: {len(indices)} final detections\n")  # Commented
            for i in indices.flatten():
                results.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_name': 'paper'
                })
                # Commented detailed prints
                # print(f"   Detection #{len(results)}:")
                # print(f"      Confidence: {confidences[i]:.3f} ({confidences[i]*100:.1f}%)")
                # print(f"      Box: x={boxes[i][0]}, y={boxes[i][1]}, w={boxes[i][2]}, h={boxes[i][3]}")
        
        return results

# ===========================
# MAIN TEST
# ===========================
def main():
    print("="*60)
    print("NCNN YOLO MODEL TESTER")
    print("="*60)
    
    # Check files exist
    print(f"\nCurrent directory: {os.getcwd()}\n")
    
    if not os.path.exists(MODEL_PARAM):
        print(f"❌ Model param file not found: {MODEL_PARAM}")
        print(f"   Available files: {os.listdir('.')}")
        return
    
    if not os.path.exists(MODEL_BIN):
        print(f"❌ Model bin file not found: {MODEL_BIN}")
        return
    
    print(f"✅ Found {MODEL_PARAM}")
    print(f"✅ Found {MODEL_BIN}\n")
    
    # Load model
    print("Loading model...")
    detector = YOLOv5Detector(MODEL_PARAM, MODEL_BIN, 
                             target_size=TARGET_SIZE, 
                             conf_threshold=CONF_THRESHOLD)
    
    print("\n" + "="*60)
    print("TESTING WITH IMAGE")
    print("="*60 + "\n")
    
    # Test with image
    if os.path.exists(TEST_IMAGE):
        print(f"Loading test image: {TEST_IMAGE}")
        img = cv2.imread(TEST_IMAGE)
        
        if img is None:
            print("❌ Failed to load image!")
            return
        
        print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]} pixels\n")
        
        # Run detection
        print("Running detection...")
        start_time = time.time()
        detections = detector.detect(img)
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Inference time: {elapsed*1000:.1f}ms")
        print(f"📊 FPS: {1/elapsed:.1f}")
        
        # Draw results
        display = img.copy()
        for det in detections:
            x, y, w, h = det['box']
            conf = det['confidence']
            
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"paper: {conf*100:.1f}%"
            cv2.putText(display, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show result
        cv2.imshow("Detection Result", display)
        print(f"\n✅ Showing result window. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print(f"   Please provide a test image or use webcam test below.\n")
    
    # Test with webcam
    print("\n" + "="*60)
    print("WEBCAM TEST (Press 'q' to quit, 's' to save frame)")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("✅ Webcam opened. Testing detection in real-time...\n")
    
    # CREATE WINDOW FIRST - This is important!
    cv2.namedWindow("Webcam Test", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    total_time = 0
    detections = []  # Store last detection results
    detect_every = 3  # Only run detection every 3 frames for speed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Only run detection every N frames (for speed)
        if frame_count % detect_every == 0:
            start_time = time.time()
            detections = detector.detect(frame)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Print detection info
            if len(detections) > 0:
                print(f"Frame {frame_count}: {len(detections)} detections in {elapsed*1000:.0f}ms")
        
        # Draw results (using last detection)
        display = frame.copy()
        for det in detections:
            x, y, w, h = det['box']
            conf = det['confidence']
            
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"paper: {conf*100:.1f}%"
            cv2.putText(display, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show FPS and stats
        avg_fps = frame_count / (total_time + 0.001)
        cv2.putText(display, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"Detections: {len(detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # SHOW THE FRAME - This is the key part!
        cv2.imshow("Webcam Test", display)
        
        # Wait for key press (1ms)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✅ Quitting...")
            break
        elif key == ord('s'):
            filename = f"detection_{frame_count}.jpg"
            cv2.imwrite(filename, display)
            print(f"💾 Saved frame to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test completed!")
    if total_time > 0:
        avg_fps = frame_count / total_time
        print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Total frames: {frame_count}")

if __name__ == "__main__":
    main()