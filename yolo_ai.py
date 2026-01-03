import ncnn
import cv2
import numpy as np

# Load model
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 4

net.load_param("model.ncnn.param")
net.load_model("model.ncnn.bin")

# Open camera or video
cap = cv2.VideoCapture(0)  # or your video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    input_size = 640
    
    # Preprocess - OpenCV uses BGR by default
    mat_in = ncnn.Mat.from_pixels_resize(
        frame, 
        ncnn.Mat.PixelType.PIXEL_BGR,  # Changed this line
        w, h, 
        input_size, input_size
    )
    
    norm_vals = [1/255.0, 1/255.0, 1/255.0]
    mat_in.substract_mean_normalize([], norm_vals)
    
    # Inference
    ex = net.create_extractor()
    ex.input("images", mat_in)
    
    ret, mat_out = ex.extract("output")
    
    # Display
    cv2.imshow("YOLOv5", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()