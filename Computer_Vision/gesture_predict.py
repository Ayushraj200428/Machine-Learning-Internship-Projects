import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "gesture_model.keras")
IMG_H, IMG_W = 64, 64

CLASS_NAMES = [
    "Palm", "L", "Fist", "Fist Moved",
    "Thumb", "Index", "OK", "Palm Moved",
    "C", "Down"
]

COLOR_GREEN  = (0, 220, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,   0)
COLOR_YELLOW = (0, 215, 255)


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at:\n  {MODEL_PATH}")
        print("Please run hand_gesture_recognition.py first to train and save the model.")
        sys.exit(1)
    print(f"Loading model from  →  {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!\n")
    return model


# ── Preprocess one OpenCV frame / image for the model ─────────────────────────
def preprocess(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    resized = cv2.resize(gray, (IMG_W, IMG_H))          
    norm    = resized.astype("float32") / 255.0         
    return norm[np.newaxis, ..., np.newaxis]           


# ── Run prediction and return label + confidence ──────────────────────────────
def predict(model, frame):
    inp   = preprocess(frame)
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs


# ── Draw a nice overlay on the frame ─────────────────────────────────────────
def draw_overlay(frame, label, confidence, probs):
    h, w = frame.shape[:2]

    # ── Top banner ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), COLOR_BLACK, -1)
    cv2.putText(frame,
                f"Gesture: {label}",
                (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_GREEN, 2)
    cv2.putText(frame,
                f"{confidence:.1f}%",
                (w - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_YELLOW, 2)

    # ── Probability bar chart (bottom of frame) ──────────────────────────────
    bar_area_h = 160
    bar_x0     = 10
    bar_y0     = h - bar_area_h
    bar_max_w  = w - 20
    bar_h      = 13
    gap        = 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y0 - 20), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        y = bar_y0 + i * (bar_h + gap)
        filled_w = int(prob * bar_max_w)

        cv2.rectangle(frame, (bar_x0, y),
                      (bar_x0 + bar_max_w, y + bar_h), (60, 60, 60), -1)
      
        color = COLOR_GREEN if name == label else (100, 180, 100)
        cv2.rectangle(frame, (bar_x0, y),
                      (bar_x0 + filled_w, y + bar_h), color, -1)
       
        cv2.putText(frame,
                    f"{name:<14} {prob*100:4.1f}%",
                    (bar_x0 + bar_max_w + 5, y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1)

    # ── Controls hint ────────────────────────────────────────────────────────
    cv2.putText(frame, "Q: quit   S: save screenshot",
                (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return frame


# ── MODE 1: Predict on a single image file ────────────────────────────────────
def predict_image(model, image_path):
    image_path = image_path.strip().strip('"').strip("'")

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] OpenCV could not read the image: {image_path}")
        return

    label, confidence, probs = predict(model, frame)

    print(f"\n{'─'*45}")
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Gesture   : {label}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'─'*45}")
    print("  All probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        bar = "█" * int(p * 25)
        print(f"    {name:<14} {p*100:5.1f}%  {bar}")
    print()

  
    disp = frame.copy()
    disp_h, disp_w = disp.shape[:2]
  
    min_h = 350
    if disp_h < min_h:
        scale  = min_h / disp_h
        disp_w = int(disp_w * scale)
        disp_h = min_h
        disp   = cv2.resize(disp, (disp_w, disp_h))

    annotated = draw_overlay(disp, label, confidence, probs)

    cv2.imshow("Gesture Prediction — press any key to close", annotated)

    out_path = os.path.join(BASE_DIR, "predicted_output.png")
    cv2.imwrite(out_path, annotated)
    print(f"  Annotated image saved → {out_path}\n")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── MODE 2: Live webcam prediction ────────────────────────────────────────────
def predict_webcam(model):
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Make sure it is connected.")
        return

    print("\nWebcam opened. Show your hand gesture in front of the camera.")
    print("  Q → quit       S → save screenshot\n")

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        h, w   = frame.shape[:2]
        roi_x1 = w // 2 - 120
        roi_y1 = h // 2 - 120
        roi_x2 = w // 2 + 120
        roi_y2 = h // 2 + 120

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]  

        label, confidence, probs = predict(model, roi)

        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), COLOR_GREEN, 2)
        cv2.putText(frame, "Place hand here",
                    (roi_x1, roi_y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GREEN, 1)

        frame = draw_overlay(frame, label, confidence, probs)

        cv2.imshow("Live Gesture Recognition  |  Q: quit  S: save", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: 
            break
        elif key == ord('s'):
            screenshot_count += 1
            fname = os.path.join(BASE_DIR, f"screenshot_{screenshot_count:03d}.png")
            cv2.imwrite(fname, frame)
            print(f"  Screenshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ── Main menu ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("   Hand Gesture Recognition — OpenCV Predictor")
    print("=" * 50)

    model = load_model()

    print("Choose input mode:")
    print("  1  →  Predict from an image file")
    print("  2  →  Live webcam prediction")
    print()

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    if choice == "1":
        print("\nEnter the full path to the image file.")
        print("Example: C:\\Users\\You\\Desktop\\hand.png")
        print()
        image_path = input("Image path: ").strip()
        predict_image(model, image_path)

    elif choice == "2":
        predict_webcam(model)


if __name__ == "__main__":
    main()
