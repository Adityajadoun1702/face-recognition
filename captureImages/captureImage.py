import cv2
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
TOTAL_PHOTOS = 200
DELAY_SECONDS = 0.2  # Delay between shots (set to 0 for maximum speed)
CAMERA_INDEX = 0     # 0 is default webcam; change to 1 or 2 for external USB cameras

# Create a dedicated timestamped folder inside a main "Captures" directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = os.path.join("Captures", f"Batch_{timestamp}")
os.makedirs(output_folder, exist_ok=True)

# Initialize camera using DirectShow for full resolution control on Windows
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Request ultra-high resolution; OpenCV snaps to your camera's true max resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 7680)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4320)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera initialized at: {actual_width}x{actual_height}")
print(f"Saving {TOTAL_PHOTOS} photos to: {os.path.abspath(output_folder)}")
print("Press 'Q' on the preview window to stop early.\n")

# Create a resizable preview window so 1080p/4K feeds fit nicely on your screen
window_name = "Live Capture Preview (Press Q to Quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

# Warm up camera sensor (2 seconds) while showing live preview so you can frame your shot
warmup_start = time.time()
while time.time() - warmup_start < 2.0:
    ret, frame = cap.read()
    if ret:
        preview = frame.copy()
        cv2.putText(preview, "Get ready...", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow(window_name, preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Main capture loop
count = 0
last_capture_time = 0

while count < TOTAL_PHOTOS:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    current_time = time.time()

    # Save clean, unedited full-resolution frame when delay interval has passed
    if current_time - last_capture_time >= DELAY_SECONDS:
        count += 1
        filename = os.path.join(output_folder, f"image_{count:02d}.jpg")
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"[{count}/{TOTAL_PHOTOS}] Saved {filename}")
        last_capture_time = current_time

    # Draw status text on a copy so the saved photos stay completely clean
    preview = frame.copy()
    cv2.putText(preview, f"Captured: {count}/{TOTAL_PHOTOS}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow(window_name, preview)

    # Refresh window and allow quitting early with 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nStopped early by user.")
        break

cap.release()
cv2.destroyAllWindows()
print("\nCapture complete!")
