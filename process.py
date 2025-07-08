import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# Load mô hình helmet
helmet_model = YOLO("best.pt")
helmet_model.eval()


# ==== ⚙️ CONFIG ====
ANCHOR_IMAGE_PATH = "anchors_cropped/anchor_019.jpg"
OUTPUT_FOLDER = "check"
RESIZED_SHAPE = (512, 1024)
THRESHOLD = 0.75

# ==== 📂 INIT ====
os.makedirs(f"{OUTPUT_FOLDER}/anchor", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/test", exist_ok=True)

# ==== 🧠 MODEL ====
model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                          input_shape=(224, 224, 3), pooling='avg')


def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)


# ==== 📦 LABELS + COLOR ====
labels = ["nametag", "shirt", "pants", "left_glove", "right_glove",
          "left_shoe", "right_shoe", "left_arm", "right_arm"]
colors = {"pass": (0, 255, 0), "fail": (0, 0, 255), "missing": (128, 128, 128)}


# ===NAMETAG===

def detect_nametag_better(image_path, bright_threshold=170, ratio_thresh=0.03, area_thresh=300, show=True):
    if not os.path.exists(image_path):
        print("❌ File không tồn tại:", image_path)
        return "missing", None

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold để tìm vùng sáng (thẻ tên)
    _, binary = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)

    # Tìm contours để xác định vùng sáng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_ratio = 0.0  # Khởi tạo white_ratio
    largest_area = 0
    best_box = None
    found = False

    # Tìm contour lớn nhất
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(cnt)
            best_box = (x, y, x+w, y+h)
            found = area > area_thresh

    # Tính tỷ lệ pixel sáng dựa trên contour lớn nhất
    if largest_area > 0.5:
        white_ratio = largest_area / binary.size
        print(f"🔍 Bright pixel ratio (largest cluster): {white_ratio:.2%}")

    if show and found:
        cv2.rectangle(img, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
        cv2.putText(img, f"Area: {int(largest_area)}", (best_box[0], best_box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return ("pass" if (white_ratio > ratio_thresh or found) else "fail"), best_box

def evaluate_shirt_color_hsv_direct(img, save_path=None):
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))

    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([3, 80, 80])
    upper_orange = np.array([25, 255, 255])

    # ROI giống như demo.py: vùng giữa ngực nơi hay có sọc cam
    top = int(h_img * 0.18)
    bottom = int(h_img * 0.42)
    left = int(w_img * 0.05)
    right = int(w_img * 0.95)
    roi = hsv[top:bottom, left:right]

    kernel = np.ones((5, 5), np.uint8)
    roi_mask = cv2.inRange(roi, lower_orange, upper_orange)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

    ref_top = np.array([38, 34, 247])
    ref_cam = np.array([3, 220, 231])
    ref_bottom = np.array([115, 141, 86])

    def is_similar_v2(color1, color2, h_thresh=25, s_thresh=130, v_thresh=130):
        h1, s1, v1 = color1
        h2, s2, v2 = color2
        return abs(h1 - h2) < h_thresh and abs(s1 - s2) < s_thresh and abs(v1 - v2) < v_thresh

    orange_range = (np.array([3, 40, 80]), np.array([30, 255, 255]))  # CAM linh hoạt hơn
    be_range = (np.array([5, 10, 100]), np.array([75, 80, 255]))
    blue_range = (np.array([95, 30, 35]), np.array([135, 255, 255]))  # XANH dương mở rộng

    def in_range(color, color_range):
        lower, upper = color_range
        return np.all(color >= lower) and np.all(color <= upper)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if save_path is not None:
            debug_img = img.copy()
            cv2.putText(debug_img, "No orange contour detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imwrite(save_path, debug_img)
        return "fail"

    largest_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_cnt) < 150:
        if save_path is not None:
            debug_img = img.copy()
            cv2.putText(debug_img, "Contour too small", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imwrite(save_path, debug_img)
        return "fail"

    x, y, w_box, h_box = cv2.boundingRect(largest_cnt)
    x_abs = x + left
    y_abs = y + top

    cam_roi = hsv[y_abs:y_abs + h_box, x_abs:x_abs + w_box]
    cam_mask = cv2.inRange(cam_roi, lower_orange, upper_orange)
    cam_mean = np.array(cv2.mean(cam_roi, mask=cam_mask)[:3])

    top_hsv = hsv[0:y_abs, x_abs:x_abs + w_box]
    top_mean = np.array(cv2.mean(top_hsv)[:3])

    bot_hsv = hsv[y_abs + h_box:, x_abs:x_abs + w_box]
    bot_mean = np.array(cv2.mean(bot_hsv)[:3])

    # cam_match = is_similar_v2(cam_mean, ref_cam, h_thresh=20, s_thresh=130, v_thresh=130)
    # top_match = is_similar_v2(top_mean, ref_top, h_thresh=25, s_thresh=130, v_thresh=130)
    # bottom_match = is_similar_v2(bot_mean, ref_bottom, h_thresh=25, s_thresh=130, v_thresh=130)
    cam_match = in_range(cam_mean, orange_range)
    top_match = in_range(top_mean, be_range)
    bottom_match = in_range(bot_mean, blue_range)

    result = "pass" if cam_match and top_match and bottom_match else "fail"

    if save_path is not None:
        debug_img = img.copy()

        # Vẽ vùng CAM
        cv2.rectangle(debug_img, (x_abs, y_abs), (x_abs + w_box, y_abs + h_box), (0, 165, 255), 2)
        cv2.putText(debug_img, "CAM", (x_abs, y_abs - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 165, 255), 1)

        # Vẽ vùng BE (top)
        cv2.rectangle(debug_img, (x_abs, 0), (x_abs + w_box, y_abs), (0, 255, 255), 2)
        cv2.putText(debug_img, "BE", (x_abs, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

        # Vẽ vùng XANH (bottom)
        cv2.rectangle(debug_img, (x_abs, y_abs + h_box), (x_abs + w_box, h_img), (255, 0, 0), 2)
        cv2.putText(debug_img, "BLUE", (x_abs, y_abs + h_box + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)

        # Nếu có vùng sai → ghi thông báo cụ thể
        if result == "fail":
            if not cam_match:
                cv2.putText(debug_img, "❌ Sai CAM", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not top_match:
                cv2.putText(debug_img, "❌ Sai BE (Tren)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not bottom_match:
                cv2.putText(debug_img, "❌ Sai BLUE (Duoi)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(debug_img, "✅ Dung mau dong phuc", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imwrite(save_path, debug_img)

    return result
# ==== 📌 POSE CROP ====
def crop_pose(image_path, save_folder):
    image = cv2.imread(image_path)
    image = cv2.resize(image, RESIZED_SHAPE)
    h, w, _ = image.shape
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        print(f"❌ Không phát hiện người trong ảnh: {image_path}")
        return {}, {}, image

    landmarks = results.pose_landmarks.landmark

    def get_point(lm):
        return int(lm.x * w), int(lm.y * h)

    crops = {}
    crop_paths = {}

    def save_crop(label, x1, y1, x2, y2):
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            crop = image[y1:y2, x1:x2]
            path = os.path.join(save_folder, f"crop_{label}.jpg")
            cv2.imwrite(path, crop)
            crops[label] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            crop_paths[label] = path

    # Điểm landmark chính
    ls, rs = landmarks[11], landmarks[12]
    lw, rw = landmarks[15], landmarks[16]
    la, ra = landmarks[27], landmarks[28]
    lh, rh = landmarks[23], landmarks[24]

    # Nametag
    x1, y1 = get_point(ls)
    x2, y2 = get_point(rs)
    cx1 = int((x1 + x2) * 0.5)
    cx2 = max(x1, x2) + 20
    cy1 = int((y1 + y2) * 0.5 + 0.08 * h)-20
    cy2 = cy1 + 100 #độ dài
    save_crop("nametag", cx1, cy1, cx2, cy2)

    # Găng tay
    def crop_hand(label, ids):
        pts = [landmarks[i] for i in ids]
        xs = [int(p.x * w) for p in pts]
        ys = [int(p.y * h) for p in pts]
        margin_x, margin_y = 30, 50
        save_crop(label, min(xs) - margin_x, min(ys) - margin_y, max(xs) + margin_x, max(ys) + margin_y)

    crop_hand("left_glove", [15, 17, 19, 21])
    crop_hand("right_glove", [16, 18, 20, 22])

    # Giày
    for label, pt in zip(["left_shoe", "right_shoe"], [la, ra]):
        px, py = get_point(pt)
        save_crop(label, px - 50, py - 20, px + 50, py + 60)

    # Áo
    x_ls, y_ls = get_point(ls)
    x_rs, y_rs = get_point(rs)
    shirt_x1 = min(x_ls, x_rs) - 20
    shirt_y1 = min(y_ls, y_rs) - 40
    shirt_x2 = max(x_ls, x_rs) + 20
    shirt_y2 = int((lh.y + rh.y) / 2 * h)
    save_crop("shirt", shirt_x1, shirt_y1, shirt_x2, shirt_y2)

    # Quần
    lx, ly = get_point(lh)
    rx, ry = get_point(rh)
    ankle_y = max(get_point(la)[1], get_point(ra)[1])
    save_crop("pants", min(lx, rx) - 80, min(ly, ry), max(lx, rx) + 80, ankle_y + 40)

    # Cánh tay
    for label, shoulder, wrist in zip(["left_arm", "right_arm"], [ls, rs], [lw, rw]):
        sx, sy = get_point(shoulder)
        wx, wy = get_point(wrist)
        save_crop(label, min(sx, wx) - 30, min(sy, wy) - 30, max(sx, wx) + 30, max(sy, wy) + 30)


    # === CROP VÙNG ĐẦU (HELMET) ===
    head_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # mũi, mắt, miệng, tai
    head_points = [get_point(landmarks[i]) for i in head_ids]
    xs = [p[0] for p in head_points]
    ys = [p[1] for p in head_points]

    # Mở rộng vùng đầu để lấy cả nón
    margin_x, margin_y = 40, 60
    x1, y1 = max(min(xs) - margin_x, 0), max(min(ys) - margin_y, 0)
    x2, y2 = min(max(xs) + margin_x, w), min(max(ys) + margin_y, h)

    save_crop("helmet", x1, y1, x2, y2)


    return crops, crop_paths, image, landmarks
#====Nut ÁO====
def detect_buttons_on_shirt(image_path, debug_path=None):
    CLAHE_clip = 10
    Blur_k = 2
    Canny_T1 = 53
    Canny_T2 = 150
    Area_min = 42
    Area_max = 500
    Circ_min = 0.50
    Circ_max = 1.40

    img = cv2.imread(image_path)
    if img is None:
        return "missing", {}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_clip, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    blur = cv2.GaussianBlur(clahe_img, (Blur_k * 2 + 1, Blur_k * 2 + 1), 0)
    edges = cv2.Canny(blur, Canny_T1, Canny_T2)

    left_count = 0
    right_count = 0
    result_img = img.copy()
    button_boxes = {}
    idx = 1  # để tạo nhãn shirt_buttons_1, _2,...

    for (x, y, ww, hh) in [(0, 0, w // 2, h), (w // 2, 0, w // 2, h)]:
        roi = clahe_img[y:y+hh, x:x+ww]
        roi_edges = cv2.Canny(roi, Canny_T1, Canny_T2)
        contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if Area_min < area < Area_max:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if Circ_min < circularity < Circ_max:
                    x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    cx_center = x + x_cnt + w_cnt // 2
                    if cx_center < w // 2:
                        left_count += 1
                    else:
                        right_count += 1

                    x1 = x + x_cnt
                    y1 = y + y_cnt
                    x2 = x1 + w_cnt
                    y2 = y1 + h_cnt
                    button_boxes[f"shirt_buttons_{idx}"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    idx += 1

                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    result = "pass" if left_count >= 1 and right_count >= 1 else "fail"

    if debug_path:
        status = "✅ ĐỦ NÚT" if result == "pass" else "❌ THIẾU NÚT"
        cv2.putText(result_img, f"{status} | Trái: {left_count} - Phải: {right_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(debug_path, result_img)

    return result, button_boxes

#====Mau Ao====
def extract_shirt_colors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (300, 300))  # đảm bảo tỷ lệ chuẩn

    h = img.shape[0]
    top = img[0:int(h/3), :, :]
    mid = img[int(h/3):int(2*h/3), :, :]
    bot = img[int(2*h/3):, :, :]

    color_top = np.mean(top.reshape(-1, 3), axis=0)
    color_mid = np.mean(mid.reshape(-1, 3), axis=0)
    color_bot = np.mean(bot.reshape(-1, 3), axis=0)

    return {
        "top": color_top,
        "mid": color_mid,
        "bot": color_bot
    }
def intersect_with_leg_line(box, knee, ankle):
    """
    Kiểm tra xem bounding box có cắt qua đường thẳng từ đầu gối đến gót chân không.

    Args:
        box: tuple (x1, y1, x2, y2) – toạ độ vùng da
        knee: tuple (x, y) – toạ độ đầu gối
        ankle: tuple (x, y) – toạ độ gót chân

    Returns:
        True nếu cắt qua, False nếu nằm lệch ngoài
    """
    x1, y1, x2, y2 = box
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Tọa độ điểm đầu và cuối đường trục chân
    x_knee, y_knee = knee
    x_ankle, y_ankle = ankle

    # Duyệt theo chiều y, kiểm tra từng điểm trên đường trục
    for alpha in np.linspace(0, 1, 20):  # kiểm tra 20 điểm trên đoạn thẳng
        x_line = int((1 - alpha) * x_knee + alpha * x_ankle)
        y_line = int((1 - alpha) * y_knee + alpha * y_ankle)

        if x_min <= x_line <= x_max and y_min <= y_line <= y_max:
            return True  # Có giao

    return False  # Không cắt qua

# ==DeLoy==
def run_inference(test_image_path):
    # Tạo lại thư mục kết quả
    os.makedirs(f"{OUTPUT_FOLDER}/anchor", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/test", exist_ok=True)
    box_errors = []

    # ==== 📌 POSE CROP ====
    print("🔧 Đang crop ảnh chuẩn...")
    anchor_boxes, anchor_paths, _, _ = crop_pose(ANCHOR_IMAGE_PATH, f"{OUTPUT_FOLDER}/anchor")

    print("🔧 Đang crop ảnh test...")
    test_boxes, test_paths, test_image, test_landmarks = crop_pose(test_image_path, f"{OUTPUT_FOLDER}/test")


    results = {}
    early_fail = False
    all_labels = labels.copy()

    for label in all_labels:
        if label in ["left_arm", "right_arm"]:
            continue  # bỏ kiểm tra tay áo ở bước này, sẽ kiểm tra sau nếu shirt pass
        if label in ["left_glove", "right_glove"]:
            path = test_paths.get(label)
            if path is None:
                result = "missing"
            else:
                img = cv2.imread(path)
                if img is None:
                    result = "missing"
                else:
                    # === Kiểm tra toàn bàn tay
                    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask_full = cv2.inRange(hsv_full, np.array([0, 20, 70], dtype=np.uint8),
                                                        np.array([20, 255, 255], dtype=np.uint8))
                    skin_ratio_full = np.sum(mask_full == 255) / mask_full.size
                    print(f"[{label.upper()}] skin ratio (full): {skin_ratio_full:.2%}")

                    # === Kiểm tra đầu ngón tay (1/3 dưới)
                    h = img.shape[0]
                    roi = img[int(h * 2 / 3):, :]
                    hsv_tip = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_tip = cv2.inRange(hsv_tip, np.array([0, 20, 70], dtype=np.uint8),
                                                    np.array([20, 255, 255], dtype=np.uint8))
                    skin_ratio_tip = np.sum(mask_tip == 255) / mask_tip.size
                    print(f"[{label.upper()}] skin ratio (fingertips): {skin_ratio_tip:.2%}")

                    # === Tổng hợp kết luận
                    if skin_ratio_full > 0.4:
                        result = "fail"
                        if label in test_boxes:
                            box = test_boxes[label]
                            box_errors.append({
                                "label": f"{label}_no_glove",
                                "box": (box["x1"], box["y1"], box["x2"], box["y2"]),
                                "color": (0, 0, 255)
                            })
                    elif skin_ratio_tip > 0.02:
                        result = "fail"
                        if label in test_boxes:
                            contours, _ = cv2.findContours(mask_tip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                if cv2.contourArea(cnt) < 100:
                                    continue
                                x, y, w, h = cv2.boundingRect(cnt)
                                box = test_boxes[label]
                                x1 = box["x1"] + x
                                y1 = box["y1"] + int((box["y2"] - box["y1"]) * 2 / 3) + y
                                x2 = x1 + w
                                y2 = y1 + h
                                box_errors.append({
                                    "label": f"{label}_tip_skin",
                                    "box": (x1, y1, x2, y2),
                                    "color": (0, 0, 255)
                                })
                    else:
                        result = "pass"

            results[label] = result
        if label == "nametag":
            if early_fail:
                results[label] = "fail"
                continue
            result, nametag_box = detect_nametag_better(test_paths.get(label))
            offset = test_boxes["nametag"]
            if nametag_box:
                x1_crop, y1_crop, x2_crop, y2_crop = nametag_box
                x1 = offset["x1"] + x1_crop
                y1 = offset["y1"] + y1_crop
                x2 = offset["x1"] + x2_crop
                y2 = offset["y1"] + y2_crop
            results[label] = result

        else:
            if label == "shirt":
                shirt_path = test_paths.get("shirt")  # ảnh đã crop
                result = "missing"
                if shirt_path is not None:
                    shirt_img = cv2.imread(shirt_path)
                    if shirt_img is not None:
                        debug_path = os.path.join(OUTPUT_FOLDER, "test", "shirt_debug.jpg")
                        result = evaluate_shirt_color_hsv_direct(shirt_img, save_path=debug_path)

                        # ✅ Kiểm tra nút áo ngay sau khi kiểm tra màu áo
                        button_result, button_boxes = detect_buttons_on_shirt(
                            shirt_path,
                            debug_path=os.path.join(OUTPUT_FOLDER, "test", "button_debug.jpg")
                        )
                        results["shirt_buttons"] = button_result
                        all_labels.extend(button_boxes.keys())
                        test_boxes.update(button_boxes)

            if label in ["shirt", "pants"] and result == "fail":
                early_fail = True
                # Nếu pants là pass, kiểm tra xem có bị sắn (lộ da) không
            if label == "pants":
                path = test_paths.get("pants")
                img = cv2.imread(path) if path else None
                if img is not None:
                    result = "pass"
                else:
                    result = "missing"

                path = test_paths.get("pants")
                if path is not None:
                    img = cv2.imread(path)
                    h = img.shape[0]

                    start_row = int(h * 1 / 2)  # kiểm tra từ nửa dưới quần
                    lower_part = img[start_row:, :]
                    cv2.imwrite("debug_lower_pants.jpg", lower_part)

                    hsv = cv2.cvtColor(lower_part, cv2.COLOR_BGR2HSV)
                    lower = np.array([0, 20, 70], dtype=np.uint8)
                    upper = np.array([20, 255, 255], dtype=np.uint8)
                    mask = cv2.inRange(hsv, lower, upper)
                    cv2.imwrite("debug_mask_pants.jpg", mask)

                    skin_ratio = np.sum(mask == 255) / mask.size
                    print(f"[PANTS SẮN] Skin ratio (lower): {skin_ratio:.2%}")

                    if skin_ratio > 0.02:
                        # ======= BỔ SUNG: In ra vị trí vùng da so với đầu gối và gót chân ========
                        def get_point(lm):  # Convert landmark to pixel
                            return int(lm.x * test_image.shape[1]), int(lm.y * test_image.shape[0])

                        left_knee = get_point(test_landmarks[25])
                        right_knee = get_point(test_landmarks[26])
                        left_ankle = get_point(test_landmarks[27])
                        right_ankle = get_point(test_landmarks[28])

                        print(f"LEFT_KNEE: {left_knee}")
                        print(f"RIGHT_KNEE: {right_knee}")
                        print(f"LEFT_ANKLE: {left_ankle}")
                        print(f"RIGHT_ANKLE: {right_ankle}")

                        if label in test_boxes:
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                if cv2.contourArea(cnt) < 50:
                                    continue
                                x, y, w, h_cnt = cv2.boundingRect(cnt)
                                box = test_boxes["pants"]
                                x1 = box["x1"] + x
                                y1 = box["y1"] + start_row + y
                                x2 = x1 + w
                                y2 = y1 + h_cnt
                                region_box = (x1, y1, x2, y2)

                                # 🧠 Kiểm tra có giao với trục chân không
                                if intersect_with_leg_line(region_box, left_knee, left_ankle) or \
                                        intersect_with_leg_line(region_box, right_knee, right_ankle):
                                    print("✅ Vùng da giao với chân → là lỗi thật")
                                    test_boxes["pants_rolled_up"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                    all_labels.append("pants_rolled_up")
                                    results["pants_rolled_up"] = "fail"
                                else:
                                    print("❌ Bỏ qua vùng da không nằm trên chân")

                results["pants"] = result
        if label == "shirt" and result == "pass":
            for arm_label in ["left_arm", "right_arm"]:
                path = test_paths.get(arm_label)
                if path is None:
                    results[arm_label] = "missing"
                    continue

                img = cv2.imread(path)

                # Cắt 2/3 dưới ảnh tay để tránh vùng vai
                h = img.shape[0]
                roi = img[int(h / 3):, :]  # từ 1/3 chiều cao trở xuống

                # Xử lý HSV trên ROI
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 20, 70], dtype=np.uint8)
                upper = np.array([20, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                skin_ratio = np.sum(mask == 255) / mask.size

                print(f"[{arm_label.upper()}] skin ratio: {skin_ratio:.2%}")

                if skin_ratio > 0.01:
                    results[arm_label] = "fail"
                    if arm_label in test_boxes:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 100:
                                continue
                            x, y, w, h = cv2.boundingRect(cnt)
                            box = test_boxes[arm_label]
                            x1 = box["x1"] + x
                            y1 = box["y1"] + int(h / 3) + y
                            x2 = x1 + w
                            y2 = y1 + h
                            box_errors.append({
                                "label": f"{arm_label}_skin",
                                "box": (x1, y1, x2, y2),
                                "color": (0, 0, 255)
                            })
                else:
                    results[arm_label] = "pass"
        results[label] = result

        # 🎨 Vẽ khung lên ảnh
        if result == "fail":
            color = colors["fail"]
            if label == "nametag" and label in test_boxes:
                box = test_boxes[label]
                cv2.rectangle(test_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
                cv2.putText(test_image, f"{label}: {result}", (box["x1"], box["y1"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif label in test_boxes:
                box = test_boxes[label]
                cv2.rectangle(test_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
                cv2.putText(test_image, f"{label}: {result}", (box["x1"], box["y1"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # 👕 Luôn vẽ nút áo nếu phát hiện (dù pass)
        if label.startswith("shirt_buttons") and label in test_boxes:
            box = test_boxes[label]
            color = (0, 255, 0)  # Xanh lá
            cv2.rectangle(test_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
            cv2.putText(test_image, f"{label}", (box["x1"], box["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ==== HELMET CHECK ====
    helmet_path = test_paths.get("helmet")
    if helmet_path is not None:
        results_helmet = helmet_model(helmet_path)[0]  # YOLO trả về list, lấy phần đầu
        names = results_helmet.names  # danh sách class names
        detected = False

        for box in results_helmet.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id].lower()
            print("Helmet Detection:", cls_name)
            if "helmet" in cls_name:
                detected = True
                break

        if detected:
            results["helmet"] = "pass"
        else:
            results["helmet"] = "fail"
            box = test_boxes.get("helmet")
            if box:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                cv2.rectangle(test_image, (x1, y1), (x2, y2), colors["fail"], 2)
                cv2.putText(test_image, "helmet: fail", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["fail"], 2)
    else:
        results["helmet"] = "missing"

    # ==== 💾 OUTPUT ====

    output_path = os.path.join(OUTPUT_FOLDER, "test_result.jpg")
    cv2.imwrite(output_path, test_image)

    return output_path, results
