import sys
import os
import csv
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QSplitter, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QMouseEvent

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from pyorbbecsdk import *
    HAS_PYORB = True
except Exception:
    HAS_PYORB = False


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = False
        self.clicked_point = None
        self.latest_depth_mm = None
        self.proj_intrinsic = None
        self.has_output = False
        self.valid_point_count = 0
        self.A1 = None
        self.A2 = None
        self.A3 = None
        self.A4 = None
        self.A5 = None
        self.transform_matrix = None
        self.B1 = None
        self.B2 = None
        self.B3 = None
        self.B4 = None
        self.new_point_count = 0
        self.rgb_saved = False
        self.SAVE_DIR = r"C:\Users\zzx\Desktop\mechanical arm\astra2\picture"
        self.OUTPUT_DIR = r"C:\Users\zzx\Desktop\UR5e-Puncture-Control\astra2\chauncimoti"

    def run(self):
        if not HAS_PYORB:
            self.status_update.emit("pyorbbecsdk 未安装，无法启动摄像头")
            return
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, OBFrameAggregateOutputMode, AlignFilter, OBStreamType
        self.running = True
        try:
            self._run_camera()
        except Exception as e:
            self.status_update.emit(f"摄像头错误: {e}")
        finally:
            self.finished_signal.emit()

    def _run_camera(self):
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, OBFrameAggregateOutputMode, AlignFilter, OBStreamType
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
            self.status_update.emit("使用硬件D2C")
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
            self.status_update.emit("使用软件D2C")

        color_intrinsic = self.get_intrinsic_dict(color_profile)
        self.proj_intrinsic = color_intrinsic

        if self.proj_intrinsic is None:
            self.status_update.emit("未获取到内参，点击时无法输出三维坐标")
        else:
            self.status_update.emit(f"内参: {self.proj_intrinsic}")

        pipeline.start(config)

        self.status_update.emit("摄像头启动，按 q 退出，左键点击输出坐标")

        try:
            while self.running:
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
                self.latest_depth_mm = depth_mm.copy()

                valid_mask = depth_mm > 0
                depth_clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM)
                depth_norm = (
                    (depth_clipped - DEPTH_MIN_MM)
                    / (DEPTH_MAX_MM - DEPTH_MIN_MM)
                    * 255.0
                ).astype(np.uint8)

                depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                depth_vis[~valid_mask] = (0, 0, 0)

                colorbar = self.create_colorbar(
                    height=h,
                    bar_width=30,
                    vmin_mm=DEPTH_MIN_MM,
                    vmax_mm=DEPTH_MAX_MM,
                    cmap=cv2.COLORMAP_JET,
                )

                color_bgr = self.frame_to_bgr(color_frame)
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
                
                if not self.rgb_saved and color_bgr is not None:
                    if not os.path.exists(self.SAVE_DIR):
                        os.makedirs(self.SAVE_DIR)
                    save_path = os.path.join(self.SAVE_DIR, "rgb_frame.png")
                    cv2.imwrite(save_path, color_bgr)
                    self.status_update.emit(f"RGB图像已保存到: {save_path}")
                    self.rgb_saved = True

                if self.clicked_point is not None:
                    cx, cy = self.clicked_point
                    if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                        cv2.circle(color_bgr, (cx, cy), 2, (0, 0, 255), -1)
                        cv2.circle(depth_vis, (cx, cy), 5, (255, 255, 255), -1)

                        if self.latest_depth_mm is not None and self.proj_intrinsic is not None:
                            if cy < self.latest_depth_mm.shape[0] and cx < self.latest_depth_mm.shape[1]:
                                z_mm = self.latest_depth_mm[cy, cx]
                                if z_mm > 0:
                                    X, Y, Z = self.pixel_to_3d(cx, cy, z_mm, self.proj_intrinsic)
                                    X_cal, Y_cal, Z_cal = self.calibrate_coordinates(X, Y, Z)
                                    if not self.has_output:
                                        if self.valid_point_count < 5:
                                            self.status_update.emit(f"像素 ({cx}, {cy}) 测量值: X={X:.2f} mm, Y={Y:.2f} mm, Z={Z:.2f} mm 校准值: X={X_cal:.2f} mm, Y={Y_cal:.2f} mm, Z={Z_cal:.2f} mm")
                                            if self.valid_point_count == 0:
                                                self.A1 = (X_cal, Y_cal, Z_cal)
                                                self.status_update.emit("已存储到 A1")
                                            elif self.valid_point_count == 1:
                                                self.A2 = (X_cal, Y_cal, Z_cal)
                                                self.status_update.emit("已存储到 A2")
                                            elif self.valid_point_count == 2:
                                                self.A3 = (X_cal, Y_cal, Z_cal)
                                                self.status_update.emit("已存储到 A3")
                                            elif self.valid_point_count == 3:
                                                self.A4 = (X_cal, Y_cal, Z_cal)
                                                self.status_update.emit("已存储到 A4")
                                            elif self.valid_point_count == 4:
                                                self.A5 = (X_cal, Y_cal, Z_cal)
                                                self.status_update.emit("已存储到 A5，已采集满5个点，正在计算变换矩阵...")
                                                self.transform_matrix = self.compute_transform_matrix()
                                            self.valid_point_count += 1
                                            self.status_update.emit(f"当前已采集 {self.valid_point_count}/5 个有效点")
                                            if self.valid_point_count == 5:
                                                self.status_update.emit("请继续点击4个点，将输出新坐标系下的坐标并保存到文件")
                                        elif self.new_point_count < 4:
                                            P_old = np.array([X_cal, Y_cal, Z_cal, 1.0])
                                            P_new = self.transform_matrix @ P_old
                                            self.status_update.emit(f"像素 ({cx}, {cy}) 原坐标系: X={X_cal:.2f} mm, Y={Y_cal:.2f} mm, Z={Z_cal:.2f} mm 新坐标系: X={P_new[0]:.2f} mm, Y={P_new[1]:.2f} mm, Z={P_new[2]:.2f} mm")
                                            if self.new_point_count == 0:
                                                self.B1 = (P_new[0], P_new[1], P_new[2])
                                                self.status_update.emit("已存储到 B1")
                                            elif self.new_point_count == 1:
                                                self.B2 = (P_new[0], P_new[1], P_new[2])
                                                self.status_update.emit("已存储到 B2")
                                            elif self.new_point_count == 2:
                                                self.B3 = (P_new[0], P_new[1], P_new[2])
                                                self.status_update.emit("已存储到 B3")
                                            elif self.new_point_count == 3:
                                                self.B4 = (P_new[0], P_new[1], P_new[2])
                                                self.status_update.emit("已存储到 B4，已采集满4个新坐标点，正在保存到文件...")
                                                self.save_points_to_file()
                                            self.new_point_count += 1
                                            self.status_update.emit(f"当前已采集 {self.new_point_count}/4 个新坐标点")
                                        else:
                                            P_old = np.array([X_cal, Y_cal, Z_cal, 1.0])
                                            P_new = self.transform_matrix @ P_old
                                            self.status_update.emit(f"像素 ({cx}, {cy}) 原坐标系: X={X_cal:.2f} mm, Y={Y_cal:.2f} mm, Z={Z_cal:.2f} mm 新坐标系: X={P_new[0]:.2f} mm, Y={P_new[1]:.2f} mm, Z={P_new[2]:.2f} mm")
                                        self.has_output = True
                                else:
                                    if not self.has_output:
                                        self.status_update.emit(f"像素 ({cx}, {cy}) 深度无效，等待有效深度...")

                depth_display = np.hstack((depth_vis, colorbar))
                combined = np.hstack((color_bgr, depth_display))

                self.frame_ready.emit(combined)

        finally:
            pipeline.stop()

    def stop(self):
        self.running = False

    def set_clicked_point(self, x, y):
        self.clicked_point = (x, y)
        self.has_output = False

    # Helper methods copied from dingwei.py
    def create_colorbar(self, height, bar_width, vmin_mm, vmax_mm, cmap=cv2.COLORMAP_JET):
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

    def frame_to_bgr(self, color_frame):
        from pyorbbecsdk import OBFormat
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
            self.status_update.emit(f"暂不支持的彩色格式: {fmt}")
            return None

    def get_intrinsic_dict(self, profile):
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

    def pixel_to_3d(self, u, v, z_mm, intrinsic):
        fx = intrinsic["fx"]
        fy = intrinsic["fy"]
        cx = intrinsic["cx"]
        cy = intrinsic["cy"]
        X = (u - cx) * z_mm / fx
        Y = (v - cy) * z_mm / fy
        Z = z_mm
        return X, Y, Z

    def calibrate_coordinates(self, x_meas, y_meas, z_meas):
        x_true = 1.00078 * x_meas + 0.02789 * z_meas + 20.66300
        y_true = 1.00148 * y_meas + (-0.03449) * z_meas + 12.25568
        z_true = 1.02314 * z_meas + (-11.14300)
        return x_true, y_true, z_true

    def compute_transform_matrix(self):
        P1 = np.array(self.A1)
        P2 = np.array(self.A2)
        P3 = np.array(self.A3)
        P4 = np.array(self.A4)
        P5 = np.array(self.A5)

        mid_point = (P1 + P2) / 2.0
        print(f"\nA1-A2中点: ({mid_point[0]:.2f}, {mid_point[1]:.2f}, {mid_point[2]:.2f})")

        vec_A1_A2 = P2 - P1
        print(f"A1到A2的向量: ({vec_A1_A2[0]:.2f}, {vec_A1_A2[1]:.2f}, {vec_A1_A2[2]:.2f})")

        y_axis = vec_A1_A2 / np.linalg.norm(vec_A1_A2)
        print(f"Y轴方向(单位向量): ({y_axis[0]:.6f}, {y_axis[1]:.6f}, {y_axis[2]:.6f})")

        v1 = P4 - P3
        v2 = P5 - P3
        z_axis = np.cross(v1, v2)
        z_axis = z_axis / np.linalg.norm(z_axis)
        if z_axis[1] > 0:
            z_axis = -z_axis
        print(f"x-y平面法向量(Z轴): ({z_axis[0]:.6f}, {z_axis[1]:.6f}, {z_axis[2]:.6f})")

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        print(f"X轴方向(单位向量): ({x_axis[0]:.6f}, {x_axis[1]:.6f}, {x_axis[2]:.6f})")

        new_origin = mid_point - 300.0 * x_axis
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

    def save_points_to_file(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

        file_path = os.path.join(self.OUTPUT_DIR, "1.csv")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(",x(mm),y(mm),z(mm)\n")
            f.write(f"B1,{self.B1[0]:.2f},{self.B1[1]:.2f},{self.B1[2]:.2f}\n")
            f.write(f"B2,{self.B2[0]:.2f},{self.B2[1]:.2f},{self.B2[2]:.2f}\n")
            f.write(f"B3,{self.B3[0]:.2f},{self.B3[1]:.2f},{self.B3[2]:.2f}\n")
            f.write(f"B4,{self.B4[0]:.2f},{self.B4[1]:.2f},{self.B4[2]:.2f}\n")

        self.status_update.emit(f"4个新坐标点已保存到: {file_path}")
        print(f"B1: ({self.B1[0]:.2f}, {self.B1[1]:.2f}, {self.B1[2]:.2f}) mm")
        print(f"B2: ({self.B2[0]:.2f}, {self.B2[1]:.2f}, {self.B2[2]:.2f}) mm")
        print(f"B3: ({self.B3[0]:.2f}, {self.B3[1]:.2f}, {self.B3[2]:.2f}) mm")
        print(f"B4: ({self.B4[0]:.2f}, {self.B4[1]:.2f}, {self.B4[2]:.2f}) mm")


class RobotWindow(QWidget):
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.capture_proc = None
        self.csv_path = os.path.join(os.path.dirname(__file__), "1.csv")
        self.points = None
        self.plot_canvas = None
        self.camera_thread = None
        self.camera_label = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('机器人控制界面')
        self.setMinimumSize(1400, 1000)
        self.resize(1600, 1200)

        # Create splitter for controls and camera
        self.splitter = QSplitter(Qt.Horizontal)

        # Left side: controls
        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout()

        self.status_label = QLabel('准备就绪')
        self.status_label.setFont(QFont('Arial', 14))
        self.controls_layout.addWidget(self.status_label)

        # 移动按钮将在开始执行路线后生成
        self.btn_start = QPushButton('移动到START_POSE')
        self.btn_start.setFont(QFont('Arial', 16))
        self.btn_start.clicked.connect(self.move_start)
        # self.controls_layout.addWidget(self.btn_start)  # 暂时不添加

        self.btn_middle = QPushButton('移动到MIDDLE_POSE')
        self.btn_middle.setFont(QFont('Arial', 16))
        self.btn_middle.clicked.connect(self.move_middle)
        self.btn_middle.setEnabled(False)
        # self.controls_layout.addWidget(self.btn_middle)  # 暂时不添加

        self.btn_target = QPushButton('移动到TARGET_POSE')
        self.btn_target.setFont(QFont('Arial', 16))
        self.btn_target.clicked.connect(self.move_target)
        self.btn_target.setEnabled(False)
        # self.controls_layout.addWidget(self.btn_target)  # 暂时不添加

        # capture / csv / plot / route controls
        self.btn_capture = QPushButton('运行采集 (dingwei)')
        self.btn_capture.setFont(QFont('Arial', 16))
        self.btn_capture.clicked.connect(self.run_capture)
        self.controls_layout.addWidget(self.btn_capture)

        self.btn_load_csv = QPushButton('加载 CSV 并显示坐标')
        self.btn_load_csv.setFont(QFont('Arial', 16))
        self.btn_load_csv.clicked.connect(self.load_csv)
        self.controls_layout.addWidget(self.btn_load_csv)

        self.btn_plot = QPushButton('绘制矩形')
        self.btn_plot.setFont(QFont('Arial', 16))
        self.btn_plot.clicked.connect(self.plot_rectangle)
        self.btn_plot.setEnabled(False)
        self.controls_layout.addWidget(self.btn_plot)

        self.btn_route = QPushButton('开始执行路线')
        self.btn_route.setFont(QFont('Arial', 16))
        self.btn_route.clicked.connect(self.start_route)
        self.controls_layout.addWidget(self.btn_route)

        controls_widget.setLayout(self.controls_layout)
        self.splitter.addWidget(controls_widget)

        # Right side: camera display
        self.camera_label = QLabel('摄像头未启动')
        self.camera_label.setFont(QFont('Arial', 14))
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 1px solid black;")
        self.camera_label.mousePressEvent = self.on_camera_click
        self.splitter.addWidget(self.camera_label)

        # Set splitter proportions
        self.splitter.setSizes([400, 1200])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

    def move_start(self):
        self.status_label.setText('正在移动到START_POSE...')
        if self.controller is None:
            self.status_label.setText('仅预览界面，无控制器')
            self.btn_middle.setEnabled(True)
            return
        try:
            self.controller.move_to_start()
            self.status_label.setText('已移动到START_POSE')
            self.btn_middle.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, '错误', str(e))
            self.status_label.setText('发生错误')

    def move_middle(self):
        self.status_label.setText('正在移动到MIDDLE_POSE...')
        if self.controller is None:
            self.status_label.setText('仅预览界面，无控制器')
            self.btn_target.setEnabled(True)
            return
        try:
            self.controller.move_to_middle()
            self.status_label.setText('已移动到MIDDLE_POSE')
            self.btn_target.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, '错误', str(e))
            self.status_label.setText('发生错误')

    def move_target(self):
        self.status_label.setText('正在移动到TARGET_POSE...')
        if self.controller is None:
            self.status_label.setText('仅预览界面，无控制器')
            return
        try:
            self.controller.move_to_target()
            self.status_label.setText('已移动到TARGET_POSE')
        except Exception as e:
            QMessageBox.critical(self, '错误', str(e))
            self.status_label.setText('发生错误')

    def run_capture(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            QMessageBox.information(self, '提示', '摄像头已在运行中')
            return
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.status_update.connect(self.status_label.setText)
        self.camera_thread.finished_signal.connect(self.on_camera_finished)
        self.camera_thread.start()
        self.status_label.setText('启动摄像头...')

    def update_frame(self, frame):
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def on_camera_click(self, event):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            # Convert click position to frame coordinates
            label_size = self.camera_label.size()
            pixmap_size = self.camera_label.pixmap().size() if self.camera_label.pixmap() else label_size
            scale_x = pixmap_size.width() / label_size.width()
            scale_y = pixmap_size.height() / label_size.height()
            x = int(event.x() * scale_x)
            y = int(event.y() * scale_y)
            # Assuming frame is 800+30+800 = 1630 wide, RGB is first 800
            if x < 800:  # Click on RGB part
                self.camera_thread.set_clicked_point(x, y)

    def on_camera_finished(self):
        self.status_label.setText('摄像头已停止')
        self.camera_thread = None

    def load_csv(self):
        # load points from chauncimoti/1.csv
        try:
            points = {}
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(self.csv_path)
            with open(self.csv_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                _ = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    label = row[0].strip()
                    if label and len(row) >= 4:
                        try:
                            x = float(row[1])
                            y = float(row[2])
                            z = float(row[3])
                            points[label] = (x, y, z)
                        except ValueError:
                            continue
            # look for B1..B4
            keys = ['B1', 'B2', 'B3', 'B4']
            if not all(k in points for k in keys):
                raise ValueError('CSV 中未包含 B1..B4 四个点')
            self.points = [points[k] for k in keys]
            self.status_label.setText(f'已加载 CSV，读取到 4 个点')
            self.btn_plot.setEnabled(HAS_MATPLOTLIB)
            if not HAS_MATPLOTLIB:
                QMessageBox.information(self, '提示', '绘图库 matplotlib 未安装，无法绘图')
        except Exception as e:
            QMessageBox.critical(self, '加载失败', str(e))
            self.status_label.setText('加载 CSV 失败')

    def plot_rectangle(self):
        if self.points is None:
            QMessageBox.information(self, '提示', '请先加载 CSV')
            return
        if not HAS_MATPLOTLIB:
            QMessageBox.critical(self, '错误', '未安装 matplotlib，无法绘图')
            return
        # Stop camera if running
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        # create or update embedded matplotlib canvas
        if self.plot_canvas is None:
            fig = Figure(figsize=(5, 4))
            self.plot_canvas = FigureCanvas(fig)
            # Replace camera label with plot canvas
            self.splitter.replaceWidget(1, self.plot_canvas)
        fig = self.plot_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        xs = [p[0] for p in self.points] + [self.points[0][0]]
        ys = [p[1] for p in self.points] + [self.points[0][1]]
        zs = [p[2] for p in self.points] + [self.points[0][2]]
        ax.plot(xs, ys, zs, marker='o')
        for i, p in enumerate(self.points, start=1):
            ax.text(p[0], p[1], p[2], f'B{i}', color='red')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        # set equal aspect (approx)
        try:
            max_range = max(
                max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
            )
            mid_x = (max(xs) + min(xs)) / 2.0
            mid_y = (max(ys) + min(ys)) / 2.0
            mid_z = (max(zs) + min(zs)) / 2.0
            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        except Exception:
            pass
        self.plot_canvas.draw()

    def start_route(self):
        # 添加移动按钮到界面
        self.controls_layout.insertWidget(1, self.btn_start)
        self.controls_layout.insertWidget(2, self.btn_middle)
        self.controls_layout.insertWidget(3, self.btn_target)
        
        # Stop camera if running
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        # create and run RobotThread to perform the full route
        self.btn_route.setEnabled(False)
        self.status_label.setText('正在启动机器人并执行路线...')
        try:
            from chauncimoti.chuancizhong import RobotController
        except Exception as e:
            QMessageBox.critical(self, '错误', f'无法导入 RobotController: {e}')
            self.status_label.setText('导入 RobotController 失败')
            self.btn_route.setEnabled(True)
            return
        # Run in QThread to avoid blocking UI
        class RobotThread(QThread):
            progress = pyqtSignal(str)
            finished = pyqtSignal()

            def run(self_inner):
                try:
                    self_inner.progress.emit('初始化 RobotController...')
                    controller = RobotController()
                    self_inner.progress.emit('移动到 START_POSE')
                    controller.move_to_start()
                    self_inner.progress.emit('移动到 MIDDLE_POSE')
                    controller.move_to_middle()
                    self_inner.progress.emit('移动到 TARGET_POSE')
                    controller.move_to_target()
                    self_inner.progress.emit('路线执行完成')
                except Exception as ex:
                    self_inner.progress.emit(f'错误: {ex}')
                finally:
                    self_inner.finished.emit()

        self.robot_thread = RobotThread()
        self.robot_thread.progress.connect(lambda s: self.status_label.setText(s))
        def on_finished():
            self.status_label.setText('机器人任务已结束')
            self.btn_route.setEnabled(True)
        self.robot_thread.finished.connect(on_finished)
        self.robot_thread.start()


# 独立运行预览界面
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotWindow()
    window.show()
    sys.exit(app.exec_())
