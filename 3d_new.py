import cv2
import numpy as np
from pyorbbecsdk import *

clicked_point = None
latest_depth_mm = None
proj_intrinsic = None


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


def on_mouse(event, x, y, flags, param):
    global clicked_point, latest_depth_mm, proj_intrinsic

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    rgb_width = param["rgb_width"]
    rgb_height = param["rgb_height"]

    if not (0 <= x < rgb_width and 0 <= y < rgb_height):
        return

    clicked_point = (x, y)

    if latest_depth_mm is None:
        print("当前没有可用深度数据")
        return

    if proj_intrinsic is None:
        print("当前没有可用内参，无法计算三维坐标")
        return

    h, w = latest_depth_mm.shape
    if x >= w or y >= h:
        print(f"点击坐标越界: ({x}, {y})")
        return

    z_mm = latest_depth_mm[y, x]
    if z_mm <= 0:
        print(f"RGB点 ({x}, {y}) 对应深度无效")
        return

    X, Y, Z = pixel_to_3d(x, y, z_mm, proj_intrinsic)
    print(
        f"像素 ({x}, {y}) -> 3D: X={X:.1f} mm, Y={Y:.1f} mm, Z={Z:.1f} mm"
    )


def main():
    global latest_depth_mm, clicked_point, proj_intrinsic

    pipeline = Pipeline()
    config = Config()

    WIDTH = 800
    HEIGHT = 600
    FPS = 30
    DEPTH_MIN_MM = 600
    DEPTH_MAX_MM = 1600

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

            if clicked_point is not None:
                cx, cy = clicked_point
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    cv2.circle(color_bgr, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.circle(depth_vis, (cx, cy), 5, (255, 255, 255), -1)

                    if latest_depth_mm is not None and proj_intrinsic is not None:
                        if cy < latest_depth_mm.shape[0] and cx < latest_depth_mm.shape[1]:
                            z_mm = latest_depth_mm[cy, cx]
                            if z_mm > 0:
                                X, Y, Z = pixel_to_3d(cx, cy, z_mm, proj_intrinsic)
                                text = f"X={X:.1f} Y={Y:.1f} Z={Z:.1f} mm"
                            else:
                                text = f"({cx},{cy}) invalid"

                            cv2.putText(
                                color_bgr,
                                text,
                                (min(cx + 10, WIDTH - 360), max(cy - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )

            depth_display = np.hstack((depth_vis, colorbar))
            combined = np.hstack((color_bgr, depth_display))

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()