import cv2
import numpy as np
from pyorbbecsdk import *


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


def main():
    pipeline = Pipeline()
    config = Config()

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(
        800, 600, OBFormat.Y16, 30
    )
    config.enable_stream(depth_profile)

    config.set_frame_aggregate_output_mode(
        OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
    )

    pipeline.start(config)

    print("按 q 退出")

    DEPTH_MIN_MM = 300
    DEPTH_MAX_MM = 3000

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            depth_frame = frames.get_depth_frame()
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

            valid_mask = depth_mm > 0

            depth_clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM)

            depth_norm = ((depth_clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255.0)
            depth_norm = depth_norm.astype(np.uint8)

            depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            depth_vis[~valid_mask] = (0, 0, 0)

            colorbar = create_colorbar(
                height=h,
                bar_width=30,
                vmin_mm=DEPTH_MIN_MM,
                vmax_mm=DEPTH_MAX_MM,
                cmap=cv2.COLORMAP_JET,
            )

            display = np.hstack((depth_vis, colorbar))

            cv2.imshow("Astra2 Depth", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()