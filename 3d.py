import cv2
import numpy as np
from pyorbbecsdk import *

clicked_point = None
latest_depth_mm = None
depth_intrinsic = None


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
        img = data.reshape((h, w, 3))
        return img

    elif fmt == OBFormat.MJPG:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img

    elif fmt == OBFormat.YUYV:
        img = data.reshape((h, w, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)

    else:
        print(f"暂不支持的彩色格式: {fmt}")
        return None


def get_intrinsic_from_profile(profile):
    """
    兼容不同 pyorbbecsdk 版本的内参读取方式。
    返回字典: {"fx":..., "fy":..., "cx":..., "cy":...}
    """
    candidates = [
        "get_intrinsic",
        "get_camera_intrinsic",
        "get_intrinsics",
    ]

    intrinsic = None
    for name in candidates:
        if hasattr(profile, name):
            try:
                intrinsic = getattr(profile, name)()
                break
            except Exception:
                pass

    if intrinsic is None:
        return None

    # 兼容不同字段名
    fx = getattr(intrinsic, "fx", None)
    fy = getattr(intrinsic, "fy", None)
    cx = getattr(intrinsic, "cx", None)
    cy = getattr(intrinsic, "cy", None)

    if None in [fx, fy, cx, cy]:
        return None

    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def pixel_to_3d(u, v, z_mm, intrinsic):
    """
    像素坐标 + 深度 -> 相机坐标系三维点
    输出单位与 z_mm 一致，这里为 mm
    """
    fx = intrinsic["fx"]
    fy = intrinsic["fy"]
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]

    X = (u - cx) * z_mm / fx
    Y = (v - cy) * z_mm / fy
    Z = z_mm
    return X, Y, Z


def on_mouse(event, x, y, flags, param):
    global clicked_point, latest_depth_mm, depth_intrinsic

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    rgb_width = param["rgb_width"]
    rgb_height = param["rgb_height"]

    # 只响应左侧 RGB 图区域
    if not (0 <= x < rgb_width and 0 <= y < rgb_height):
        return

    clicked_point = (x, y)

    if latest_depth_mm is None:
        print("当前没有可用深度数据")
        return

    if depth_intrinsic is None:
        print("当前没有可用深度相机内参，无法计算三维坐标")
        return

    h, w = latest_depth_mm.shape
    if x >= w or y >= h:
        print(f"点击坐标越界: ({x}, {y})")
        return

    z_mm = latest_depth_mm[y, x]

    if z_mm <= 0:
        print(f"RGB点 ({x}, {y}) 对应深度无效")
        return

    X, Y, Z = pixel_to_3d(x, y, z_mm, depth_intrinsic)

    print(
        f"像素 ({x}, {y}) -> 相机坐标系 3D: "
        f"X={X:.1f} mm, Y={Y:.1f} mm, Z={Z:.1f} mm"
    )


def main():
    global latest_depth_mm, clicked_point, depth_intrinsic

    pipeline = Pipeline()
    config = Config()

    WIDTH = 800
    HEIGHT = 600
    FPS = 30

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(
        WIDTH, HEIGHT, OBFormat.Y16, FPS
    )
    config.enable_stream(depth_profile)

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

    config.set_frame_aggregate_output_mode(
        OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
    )

    # 先尝试从 depth_profile 获取深度相机内参
    depth_intrinsic = get_intrinsic_from_profile(depth_profile)

    # 如果 SDK 不支持上面的读取方式，就手动填
    # depth_intrinsic = {
    #     "fx": 500.0,
    #     "fy": 500.0,
    #     "cx": 400.0,
    #     "cy": 300.0,
    # }

    if depth_intrinsic is not None:
        print("深度相机内参:")
        print(depth_intrinsic)
    else:
        print("未能自动获取深度相机内参。")
        print("请检查 SDK 接口名，或手动填写 fx, fy, cx, cy。")

    pipeline.start(config)

    window_name = "RGB + Depth"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        on_mouse,
        {"rgb_width": WIDTH, "rgb_height": HEIGHT},
    )

    print("按 q 退出")
    print("左键点击左侧 RGB 图，输出该点的三维相机坐标")

    DEPTH_MIN_MM = 600
    DEPTH_MAX_MM = 1600

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame is None:
                continue

            if depth_frame.get_format() != OBFormat.Y16:
                continue

            w = depth_frame.get_width()
            h = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((h, w))

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

                    if (
                        latest_depth_mm is not None
                        and depth_intrinsic is not None
                        and cy < latest_depth_mm.shape[0]
                        and cx < latest_depth_mm.shape[1]
                    ):
                        z_mm = latest_depth_mm[cy, cx]
                        if z_mm > 0:
                            X, Y, Z = pixel_to_3d(cx, cy, z_mm, depth_intrinsic)
                            text = f"X={X:.1f} Y={Y:.1f} Z={Z:.1f} mm"
                        else:
                            text = f"({cx},{cy}) invalid"

                        cv2.putText(
                            color_bgr,
                            text,
                            (min(cx + 10, WIDTH - 330), max(cy - 10, 25)),
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