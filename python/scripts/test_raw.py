import cv2

# Force AVFoundation to grab uncompressed frames instead of compressed blocks
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# 1. Turn off Auto Exposure (Bypasses tone-mapping/gain adjustments)
# Value 0 or 1 usually forces manual mode depending on the specific macOS driver
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 

# 2. Lock the Exposure time and Gain to a fixed value
# This stops the ISP from changing sensor noise characteristics mid-stream
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust based on your lighting
cap.set(cv2.CAP_PROP_GAIN, 100)

# 3. Explicitly request raw, uncompressed pixel format conversions
# This prevents internal conversion to JPEG/MP4 blocks before handing to Python
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 'frame' is now a raw NumPy array containing the unmanipulated pixel values
    # Feed directly to your model here
    
    cv2.imshow('Raw Stream for ML', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
