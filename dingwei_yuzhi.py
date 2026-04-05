import cv2
import numpy as np
import os
from pyorbbecsdk import *

clicked_point = None
rgb_saved = False
SAVE_DIR = r"C:\Users\zzx\Desktop\mechanical arm\astra2\picture"
latest_depth_mm = None
proj_intrinsic = None
has_output = False
valid_point_count = 0
A1 = None
A2 = None
A3 = None
transform_matrix = None
B1 = None
B2 = None
B3 = None
B4 = None
new_point_count = 0
OUTPUT_DIR = r"C:\Users\zzx\Desktop\UR5e-Puncture-Control\astra2\chauncimoti"
latest_color_bgr = None
fitted_center = None
fitted_contour = None
COLOR_TOLERANCE = 10
MIN_CONTOUR_AREA = 10
CHANNEL_MODE = "B"
CHANNEL_MAP = {"B": 0, "G": 1, "R": 2, "ALL": -1}


def region_growing(image, seed_x, seed_y, tolerance=COLOR_TOLERANCE, channel_mode=CHANNEL_MODE):
    if image is None:
        return None, None
    
    h, w = image.shape[:2]
    if seed_y >= h or seed_x >= w:
        return None, None
    
    channel_idx = CHANNEL_MAP.get(channel_mode, 0)
    
    if channel_idx >= 0:
        seed_value = float(image[seed_y, seed_x, channel_idx])
    else:
        seed_value = image[seed_y, seed_x].astype(np.float32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    
    queue = [(seed_x, seed_y, seed_value)]
    visited[seed_y, seed_x] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while queue:
        x, y, seed_val = queue.pop(0)
        
        if channel_idx >= 0:
            current_value = float(image[y, x, channel_idx])
            diff = abs(current_value - seed_val)
        else:
            current_value = image[y, x].astype(np.float32)
            diff = np.sqrt(np.sum((current_value - seed_val) ** 2))
        
        if diff <= tolerance:
            mask[y, x] = 255
            new_seed = current_value
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny, new_seed))
    
    return mask, image[seed_y, seed_x]


def get_mask_center(mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        return None, None
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy), largest_contour


def create_colorbar(height, bar_width, vmin_mm, vmax_mm, cmap=cv2.COLORMAP_JET):
    grad = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    grad = np.repeat(grad, bar_width, axis=1)
    colorbar = cv2.applyColorMap(grad, cmap)
    canvas = np.full((height, bar_width + 80, 3), 255, dtype=np.uint8)
    canvas[:, :bar_width] = colorbar
    cv2.rectangle(canvas, (0, 0), (bar_width - 1, height - 1), (0, 0, 0), 1)
    ticks = 5
    for i in range(ticks):
        y = int(i * (height - 1) / (ticks - 1))
        value = vmax_mm - i * (vmax_mm - vmin_mm) / (ticks - 1)
        cv2.line(canvas, (bar_width, y), (bar_width + 8, y), (0, 0, 0), 1)
        text = f"{int(value)} mm"
        cv2.putText(
            canvas,
            text,
            (bar_width + 12, min(y + 5, height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def frame_to_bgr(color_frame):
    if color_frame is None:
        return None

    w = color_frame.get_width()
    h = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        img = data.reshape((h, w, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return data.reshape((h, w, 3))
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.YUYV:
        img = data.reshape((h, w, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    else:
        print(f"暂不支持的彩色格式: {fmt}")
        return None


def get_intrinsic_dict(profile):
    funcs = ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]
    intrinsic = None

    for name in funcs:
        if hasattr(profile, name):
            try:
                intrinsic = getattr(profile, name)()
                break
            except Exception:
                pass

    if intrinsic is None:
        return None

    fx = getattr(intrinsic, "fx", None)
    fy = getattr(intrinsic, "fy", None)
    cx = getattr(intrinsic, "cx", None)
    cy = getattr(intrinsic, "cy", None)

    if None in [fx, fy, cx, cy]:
        return None

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


def pixel_to_3d(u, v, z_mm, intrinsic):
    fx = intrinsic["fx"]
    fy = intrinsic["fy"]
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]
    X = (u - cx) * z_mm / fx
    Y = (v - cy) * z_mm / fy
    Z = z_mm
    return X, Y, Z


def calibrate_coordinates(x_meas, y_meas, z_meas):
    """
    根据校准模型计算校正后的真实坐标
    校准模型：
    X_true = 1.00142*X_meas + 0.03006*Z_meas + 18.06510
    Y_true = 1.00202*Y_meas + -0.03565*Z_meas + 14.73138
    Z_true = 1.02601*Z_meas + -15.26776
    """
    x_true = 1.00078 * x_meas + 0.02789 * z_meas + 20.66300
    y_true = 1.00148 * y_meas + (-0.03449) * z_meas + 12.25568
    z_true = 1.02314 * z_meas + (-11.14300)
    return x_true, y_true, z_true


def on_mouse(event, x, y, flags, param):
    global clicked_point, latest_depth_mm, proj_intrinsic, has_output
    global latest_color_bgr, fitted_center, fitted_contour, CHANNEL_MODE

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    rgb_width = param["rgb_width"]
    rgb_height = param["rgb_height"]

    if not (0 <= x < rgb_width and 0 <= y < rgb_height):
        return

    clicked_point = (x, y)
    has_output = False
    fitted_center = None
    fitted_contour = None
    
    print(f"\n点击位置: ({x}, {y})，当前通道: {CHANNEL_MODE}，开始分割...")

    if latest_color_bgr is None:
        print("当前没有可用的RGB图像")
        return

    mask, click_color = region_growing(latest_color_bgr, x, y, channel_mode=CHANNEL_MODE)
    print(f"Mask像素数: {np.count_nonzero(mask)}")
    if mask is None:
        print("颜色分割失败")
        return
    
    print(f"点击位置颜色 (BGR): {click_color}")

    center, contour = get_mask_center(mask)
    if center is None:
        print("未找到有效的区域")
        return
    
    fitted_center = center
    fitted_contour = contour
    
    print(f"质心坐标: ({center[0]}, {center[1]})")

    if latest_depth_mm is None:
        print("当前没有可用深度数据")
        return

    if proj_intrinsic is None:
        print("当前没有可用内参，无法计算三维坐标")
        return

    h, w = latest_depth_mm.shape
    cx, cy = center
    if cx >= w or cy >= h:
        print(f"质心坐标越界: ({cx}, {cy})")
        return

    z_mm = latest_depth_mm[cy, cx]
    if z_mm <= 0:
        print(f"质心 ({cx}, {cy}) 对应深度无效")
        return

    X, Y, Z = pixel_to_3d(cx, cy, z_mm, proj_intrinsic)
    print(
        f"质心像素 ({cx}, {cy}) -> 3D: X={X:.1f} mm, Y={Y:.1f} mm, Z={Z:.1f} mm"
    )


def main():
    global latest_depth_mm, clicked_point, proj_intrinsic, rgb_saved, has_output, valid_point_count, A1, A2, A3, transform_matrix, B1, B2, B3, B4, new_point_count
    global latest_color_bgr, fitted_center, fitted_contour, CHANNEL_MODE

    pipeline = Pipeline()
    config = Config()

    WIDTH = 800
    HEIGHT = 600
    FPS = 30
    DEPTH_MIN_MM = 600
    DEPTH_MAX_MM = 3000

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = None
    for fmt in [OBFormat.RGB, OBFormat.BGR, OBFormat.MJPG, OBFormat.YUYV]:
        try:
            color_profile = color_profiles.get_video_stream_profile(
                WIDTH, HEIGHT, fmt, FPS
            )
            if color_profile is not None:
                break
        except Exception:
            pass

    if color_profile is None:
        raise RuntimeError("未找到可用的彩色流配置")

    config.enable_stream(color_profile)

    use_hw_d2c = False
    align_filter = None

    try:
        d2c_depth_profiles = pipeline.get_d2c_depth_profile_list(
            color_profile, OBAlignMode.HW_MODE
        )

        depth_profile = None
        try:
            depth_profile = d2c_depth_profiles.get_video_stream_profile(
                WIDTH, HEIGHT, OBFormat.Y16, FPS
            )
        except Exception:
            try:
                depth_profile = d2c_depth_profiles[0]
            except Exception:
                pass

        if depth_profile is None:
            raise RuntimeError("硬件D2C无匹配深度Profile")

        config.enable_stream(depth_profile)
        config.set_align_mode(OBAlignMode.HW_MODE)
        use_hw_d2c = True
        print("使用硬件D2C")
    except Exception:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(
            WIDTH, HEIGHT, OBFormat.Y16, FPS
        )
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        print("使用软件D2C")

    color_intrinsic = get_intrinsic_dict(color_profile)
    proj_intrinsic = color_intrinsic

    if proj_intrinsic is None:
        print("未获取到内参，点击时无法输出三维坐标")
    else:
        print(proj_intrinsic)

    pipeline.start(config)

    window_name = "RGB + Depth"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        on_mouse,
        {"rgb_width": WIDTH, "rgb_height": HEIGHT},
    )

    print("按 q 退出")
    print("按 1/2/3/4 切换通道: 1=B(默认), 2=G, 3=R, 4=ALL(全部BGR)")
    print("左键点击左侧RGB图，输出该点三维坐标")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            if align_filter is not None:
                frames = align_filter.process(frames)
                if frames is None:
                    continue
                if hasattr(frames, "as_frame_set"):
                    frames = frames.as_frame_set()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame is None or color_frame is None:
                continue

            if depth_frame.get_format() != OBFormat.Y16:
                continue

            w = depth_frame.get_width()
            h = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            depth_mm = depth_data.astype(np.float32) * scale
            latest_depth_mm = depth_mm.copy()

            valid_mask = depth_mm > 0
            depth_clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM)
            depth_norm = (
                (depth_clipped - DEPTH_MIN_MM)
                / (DEPTH_MAX_MM - DEPTH_MIN_MM)
                * 255.0
            ).astype(np.uint8)

            depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_vis[~valid_mask] = (0, 0, 0)

            colorbar = create_colorbar(
                height=h,
                bar_width=30,
                vmin_mm=DEPTH_MIN_MM,
                vmax_mm=DEPTH_MAX_MM,
                cmap=cv2.COLORMAP_JET,
            )

            color_bgr = frame_to_bgr(color_frame)
            if color_bgr is None:
                color_bgr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    color_bgr,
                    "No RGB Frame",
                    (250, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            
            latest_color_bgr = color_bgr.copy()
            
            if not rgb_saved and color_bgr is not None:
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                save_path = os.path.join(SAVE_DIR, "rgb_frame.png")
                cv2.imwrite(save_path, color_bgr)
                print(f"RGB图像已保存到: {save_path}")
                rgb_saved = True

            if clicked_point is not None and fitted_center is not None:
                cx, cy = clicked_point
                fx, fy = fitted_center
                
                if 0 <= fx < WIDTH and 0 <= fy < HEIGHT:
                    cv2.circle(color_bgr, (cx, cy), 3, (255, 0, 0), -1)
                    cv2.circle(color_bgr, (fx, fy), 5, (0, 255, 0), -1)
                    cv2.line(color_bgr, (cx, cy), (fx, fy), (0, 255, 255), 1)
                    
                    if fitted_contour is not None:
                        cv2.drawContours(color_bgr, [fitted_contour], -1, (255, 255, 0), 2)
                    
                    cv2.circle(depth_vis, (fx, fy), 5, (255, 255, 255), -1)

                    if latest_depth_mm is not None and proj_intrinsic is not None:
                        if fy < latest_depth_mm.shape[0] and fx < latest_depth_mm.shape[1]:
                            z_mm = latest_depth_mm[fy, fx]
                            if z_mm > 0:
                                X, Y, Z = pixel_to_3d(fx, fy, z_mm, proj_intrinsic)
                                X_cal, Y_cal, Z_cal = calibrate_coordinates(X, Y, Z)
                                if not has_output:
                                    if valid_point_count < 3:
                                        print(f"质心像素 ({fx}, {fy})")
                                        print(f"  测量值: X={X:.2f} mm, Y={Y:.2f} mm, Z={Z:.2f} mm")
                                        print(f"  校准值: X={X_cal:.2f} mm, Y={Y_cal:.2f} mm, Z={Z_cal:.2f} mm")
                                        print("-" * 50)
                                        if valid_point_count == 0:
                                            A1 = (X_cal, Y_cal, Z_cal)
                                            print(f"已存储到 A1")
                                        elif valid_point_count == 1:
                                            A2 = (X_cal, Y_cal, Z_cal)
                                            print(f"已存储到 A2")
                                        elif valid_point_count == 2:
                                            A3 = (X_cal, Y_cal, Z_cal)
                                            print(f"已存储到 A3")
                                            print(f"已采集满3个点，正在计算变换矩阵...")
                                        valid_point_count += 1
                                        print(f"当前已采集 {valid_point_count}/3 个有效点")
                                        if valid_point_count == 3:
                                            transform_matrix = compute_transform_matrix()
                                    elif new_point_count < 4:
                                        P_old = np.array([X_cal, Y_cal, Z_cal, 1.0])
                                        P_new = transform_matrix @ P_old
                                        print(f"\n质心像素 ({fx}, {fy})")
                                        print(f"  原坐标系: X={X_cal:.2f} mm, Y={Y_cal:.2f} mm, Z={Z_cal:.2f} mm")
                                        print(f"  新坐标系: X={P_new[0]:.2f} mm, Y={P_new[1]:.2f} mm, Z={P_new[2]:.2f} mm")
                                        print("-" * 50)
                                        if new_point_count == 0:
                                            B1 = (P_new[0], P_new[1], P_new[2])
                                            print(f"已存储到 B1")
                                        elif new_point_count == 1:
                                            B2 = (P_new[0], P_new[1], P_new[2])
                                            print(f"已存储到 B2")
                                        elif new_point_count == 2:
                                            B3 = (P_new[0], P_new[1], P_new[2])
                                            print(f"已存储到 B3")
                                        elif new_point_count == 3:
                                            B4 = (P_new[0], P_new[1], P_new[2])
                                            print(f"已存储到 B4")
                                            print(f"已采集满4个新坐标点，正在保存到文件...")
                                            save_points_to_file()
                                        new_point_count += 1
                                        print(f"当前已采集 {new_point_count}/4 个新坐标点")
                                    else:
                                        P_old = np.array([X_cal, Y_cal, Z_cal, 1.0])
                                        P_new = transform_matrix @ P_old
                                        print(f"\n质心像素 ({fx}, {fy})")
                                        print(f"  原坐标系: X={X_cal:.2f} mm, Y={Y_cal:.2f} mm, Z={Z_cal:.2f} mm")
                                        print(f"  新坐标系: X={P_new[0]:.2f} mm, Y={P_new[1]:.2f} mm, Z={P_new[2]:.2f} mm")
                                        print("-" * 50)
                                    has_output = True
                            else:
                                if not has_output:
                                    print(f"质心 ({fx}, {fy}) 深度无效，等待有效深度...")

            depth_display = np.hstack((depth_vis, colorbar))
            combined = np.hstack((color_bgr, depth_display))

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                CHANNEL_MODE = "B"
                print(f"切换到 B 通道模式")
            elif key == ord("2"):
                CHANNEL_MODE = "G"
                print(f"切换到 G 通道模式")
            elif key == ord("3"):
                CHANNEL_MODE = "R"
                print(f"切换到 R 通道模式")
            elif key == ord("4"):
                CHANNEL_MODE = "ALL"
                print(f"切换到 ALL 通道模式（使用全部BGR通道）")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def compute_transform_matrix():
    global A1, A2, A3

    P1 = np.array(A1)
    P2 = np.array(A2)
    P3 = np.array(A3)

    mid_point = (P1 + P2) / 2.0
    print(f"\nA1-A2中点: ({mid_point[0]:.2f}, {mid_point[1]:.2f}, {mid_point[2]:.2f})")

    vec_A1_A2 = P2 - P1
    print(f"A1到A2的向量: ({vec_A1_A2[0]:.2f}, {vec_A1_A2[1]:.2f}, {vec_A1_A2[2]:.2f})")

    y_axis = vec_A1_A2 / np.linalg.norm(vec_A1_A2)
    print(f"Y轴方向(单位向量): ({y_axis[0]:.6f}, {y_axis[1]:.6f}, {y_axis[2]:.6f})")

    v1 = P1 - P3
    v2 = P2 - P3
    x_axis = np.cross(v1, v2)
    x_axis = x_axis / np.linalg.norm(x_axis)
    if x_axis[2] > 0:
        x_axis = -x_axis
    print(f"平面法向量(X轴): ({x_axis[0]:.6f}, {x_axis[1]:.6f}, {x_axis[2]:.6f})")

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    print(f"Z轴方向(单位向量): ({z_axis[0]:.6f}, {z_axis[1]:.6f}, {z_axis[2]:.6f})")

    new_origin = mid_point - 307.0 * x_axis
    print(f"新坐标系原点: ({new_origin[0]:.2f}, {new_origin[1]:.2f}, {new_origin[2]:.2f})")

    R = np.array([x_axis, y_axis, z_axis])
    t = -R @ new_origin

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    print("\n" + "=" * 60)
    print("从原坐标系到新坐标系的变换矩阵 T (4x4):")
    print("=" * 60)
    print(f"[{T[0,0]:10.6f}, {T[0,1]:10.6f}, {T[0,2]:10.6f}, {T[0,3]:10.2f}]")
    print(f"[{T[1,0]:10.6f}, {T[1,1]:10.6f}, {T[1,2]:10.6f}, {T[1,3]:10.2f}]")
    print(f"[{T[2,0]:10.6f}, {T[2,1]:10.6f}, {T[2,2]:10.6f}, {T[2,3]:10.2f}]")
    print(f"[{T[3,0]:10.6f}, {T[3,1]:10.6f}, {T[3,2]:10.6f}, {T[3,3]:10.2f}]")
    print("=" * 60)

    print("\n变换矩阵含义:")
    print("  原坐标系中的点 P_old 变换到新坐标系: P_new = T @ [P_old, 1]^T")
    print("  或使用齐次坐标: [x_new, y_new, z_new, 1]^T = T @ [x_old, y_old, z_old, 1]^T")
    print("\n请继续点击4个点，将输出新坐标系下的坐标并保存到文件")

    return T


def save_points_to_file():
    global B1, B2, B3, B4, OUTPUT_DIR

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    file_path = os.path.join(OUTPUT_DIR, "1.csv")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(",x(mm),y(mm),z(mm)\n")
        f.write(f"B1,{B1[0]:.2f},{B1[1]:.2f},{B1[2]:.2f}\n")
        f.write(f"B2,{B2[0]:.2f},{B2[1]:.2f},{B2[2]:.2f}\n")
        f.write(f"B3,{B3[0]:.2f},{B3[1]:.2f},{B3[2]:.2f}\n")
        f.write(f"B4,{B4[0]:.2f},{B4[1]:.2f},{B4[2]:.2f}\n")

    print(f"\n4个新坐标点已保存到: {file_path}")
    print(f"B1: ({B1[0]:.2f}, {B1[1]:.2f}, {B1[2]:.2f}) mm")
    print(f"B2: ({B2[0]:.2f}, {B2[1]:.2f}, {B2[2]:.2f}) mm")
    print(f"B3: ({B3[0]:.2f}, {B3[1]:.2f}, {B3[2]:.2f}) mm")
    print(f"B4: ({B4[0]:.2f}, {B4[1]:.2f}, {B4[2]:.2f}) mm")


if __name__ == "__main__":
    main()