import os
import cv2
import time
import threading
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pyorbbecsdk import *

# --- 1. 辅助函数：生成纯 2D 的 Colorbar 图像 ---
def create_colorbar_image(height, bar_width, vmin_mm, vmax_mm, cmap=cv2.COLORMAP_JET):
    grad = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    grad = np.repeat(grad, bar_width, axis=1)
    colorbar = cv2.applyColorMap(grad, cmap)
    canvas = np.full((height, bar_width + 80, 3), 255, dtype=np.uint8)
    canvas[:, :bar_width] = colorbar
    cv2.rectangle(canvas, (0, 0), (bar_width - 1, height - 1), (0, 0, 0), 1)
    ticks = 6
    for i in range(ticks):
        y = int(i * (height - 1) / (ticks - 1))
        value = vmax_mm - i * (vmax_mm - vmin_mm) / (ticks - 1)
        cv2.line(canvas, (bar_width, y), (bar_width + 8, y), (0, 0, 0), 1)
        text = f"{int(value)} mm"
        cv2.putText(
            canvas, text,
            (bar_width + 12, min(y + 5, height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
    return canvas

# --- 2. 辅助函数：解码与内参 ---
def frame_to_rgb(color_frame):
    if color_frame is None: return None
    w, h = color_frame.get_width(), color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB: return data.reshape((h, w, 3))
    elif fmt == OBFormat.BGR: return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_BGR2RGB)
    elif fmt == OBFormat.MJPG:
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
    elif fmt == OBFormat.YUYV:
        img_bgr = cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUYV)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return None

def get_intrinsic_dict(profile):
    funcs = ["get_intrinsic", "get_camera_intrinsic", "get_intrinsics"]
    for name in funcs:
        if hasattr(profile, name):
            try: return getattr(profile, name)()
            except Exception: pass
    return None

# --- 3. 核心应用程序类 ---
class OrbbecGUIApp:
    def __init__(self):
        # 基础参数
        self.COLOR_WIDTH, self.COLOR_HEIGHT = 800, 600
        self.DEPTH_WIDTH, self.DEPTH_HEIGHT = 800, 600
        self.FPS = 5
        self.DEPTH_MIN_MM = 500
        self.DEPTH_MAX_MM = 3000
        self.is_running = True
        self.is_first_frame = True

        # 初始化 Open3D 应用程序
        gui.Application.instance.initialize()
        # 创建大窗口：宽 1800，高 600
        self.window = gui.Application.instance.create_window("Orbbec Unified Pro Viewer", 1800, 600)
        
        # 创建三个区域控件
        self.scene_left = gui.SceneWidget()
        self.scene_left.scene = rendering.Open3DScene(self.window.renderer)
        
        self.scene_right = gui.SceneWidget()
        self.scene_right.scene = rendering.Open3DScene(self.window.renderer)
        
        self.image_widget = gui.ImageWidget()
        
        # 将三个控件添加到窗口并设置动态布局回调
        self.window.add_child(self.scene_left)
        self.window.add_child(self.scene_right)
        self.window.add_child(self.image_widget)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        
        # --- [新增核心逻辑]：绑定 GUI 渲染时钟 ---
        self.window.set_on_tick_event(self._on_tick)

        # 绘制固定的 2D Colorbar
        cb_img_bgr = create_colorbar_image(600, 40, self.DEPTH_MIN_MM, self.DEPTH_MAX_MM)
        cb_img_rgb = cv2.cvtColor(cb_img_bgr, cv2.COLOR_BGR2RGB)
        cb_img_rgb = np.ascontiguousarray(cb_img_rgb)
        self.image_widget.update_image(o3d.geometry.Image(cb_img_rgb))

        # 材质设置 (取消光照，仅渲染点云颜色)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 2.0

        # 初始化相机
        self._init_orbbec()

        # 开启后台相机拉取线程
        self.thread = threading.Thread(target=self._update_thread)
        self.thread.start()

    def _on_layout(self, layout_context):
        # 根据窗口当前实际大小，划分布局
        r = self.window.content_rect
        cb_width = 160 # 最右侧 Colorbar 固定分配 160 像素
        scene_w = (r.width - cb_width) // 2 # 剩余空间左边和中间平分
        
        self.scene_left.frame = gui.Rect(r.x, r.y, scene_w, r.height)
        self.scene_right.frame = gui.Rect(r.x + scene_w, r.y, scene_w, r.height)
        self.image_widget.frame = gui.Rect(r.x + scene_w * 2, r.y, cb_width, r.height)

    def _on_tick(self):
        # 这个函数会以显示器的刷新率（通常 60FPS）飞速执行
        if self.is_first_frame:
            return False
            
        # 提取左右画面的相机视角矩阵
        view_left = np.asarray(self.scene_left.scene.camera.get_view_matrix())
        view_right = np.asarray(self.scene_right.scene.camera.get_view_matrix())
        
        # 只要检测到两者不同（无论是因为拖拽了左边还是右边），强行同步为左边的视角
        if not np.allclose(view_left, view_right):
            self.scene_right.scene.camera.copy_from(self.scene_left.scene.camera)
            self.window.post_redraw()
            return True # 告诉引擎界面有变动，需要渲染更新
            
        return False

    def _on_close(self):
        # 点击右上角 X 时优雅退出
        self.is_running = False
        self.thread.join()
        self.pipeline.stop()
        print("相机连接已安全关闭。")
        return True

    def _init_orbbec(self):
        self.pipeline = Pipeline()
        config = Config()

        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None
        for fmt in [OBFormat.RGB, OBFormat.MJPG, OBFormat.YUYV]:
            try:
                color_profile = color_profiles.get_video_stream_profile(self.COLOR_WIDTH, self.COLOR_HEIGHT, fmt, self.FPS)
                if color_profile is not None: break
            except Exception: pass
        if color_profile is None: raise RuntimeError("未找到彩色流配置")
        config.enable_stream(color_profile)

        self.align_filter = None
        try:
            d2c_depth_profiles = self.pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
            depth_profile = d2c_depth_profiles.get_video_stream_profile(self.DEPTH_WIDTH, self.DEPTH_HEIGHT, OBFormat.Y16, self.FPS)
            config.enable_stream(depth_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)
        except Exception:
            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_video_stream_profile(self.DEPTH_WIDTH, self.DEPTH_HEIGHT, OBFormat.Y16, self.FPS)
            config.enable_stream(depth_profile)
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        intrinsic = get_intrinsic_dict(color_profile)
        # [已修复]：使用点操作符访问内参属性
        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.COLOR_WIDTH, self.COLOR_HEIGHT, intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy
        )
        self.pipeline.start(config)

    def _update_thread(self):
        # 后台死循环：不断向相机索要数据
        while self.is_running:
            frames = self.pipeline.wait_for_frames(500)
            if frames is None: continue

            if self.align_filter is not None:
                frames = self.align_filter.process(frames)
                if frames is None: continue
                if hasattr(frames, "as_frame_set"): frames = frames.as_frame_set()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame is None or color_frame is None or depth_frame.get_format() != OBFormat.Y16:
                continue

            aligned_w, aligned_h = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((aligned_h, aligned_w))
            
            o3d_depth = o3d.geometry.Image(depth_data_raw)
            o3d_depth_scale = 1000.0 / scale 

            # 生成中间伪彩深度图
            depth_mm = depth_data_raw.astype(np.float32) * scale
            valid_mask = depth_mm > 0
            depth_clipped = np.clip(depth_mm, self.DEPTH_MIN_MM, self.DEPTH_MAX_MM)
            depth_norm = ((depth_clipped - self.DEPTH_MIN_MM) / (self.DEPTH_MAX_MM - self.DEPTH_MIN_MM) * 255.0).astype(np.uint8)
            
            depth_color_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_color_bgr[~valid_mask] = (0, 0, 0)
            o3d_pseudo_color = o3d.geometry.Image(np.ascontiguousarray(cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)))

            # 解析左侧真实彩色图
            color_image = frame_to_rgb(color_frame)
            if color_image is None: continue
            o3d_real_color = o3d.geometry.Image(np.ascontiguousarray(color_image))

            # 生成两个点云
            rgbd_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_real_color, o3d_depth, depth_scale=o3d_depth_scale, depth_trunc=4.0, convert_rgb_to_intensity=False)
            pcd_left = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_left, self.o3d_intrinsic)
            
            rgbd_right = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_pseudo_color, o3d_depth, depth_scale=o3d_depth_scale, depth_trunc=4.0, convert_rgb_to_intensity=False)
            pcd_right = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_right, self.o3d_intrinsic)

            # 翻转匹配视角
            transform_mat = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            pcd_left.transform(transform_mat)
            pcd_right.transform(transform_mat)

            # 核心机制：将计算好的数据抛回主界面的 UI 线程进行渲染更新
            def update_ui():
                self.scene_left.scene.clear_geometry()
                self.scene_left.scene.add_geometry("left", pcd_left, self.mat)
                
                self.scene_right.scene.clear_geometry()
                self.scene_right.scene.add_geometry("right", pcd_right, self.mat)
                
                if self.is_first_frame:
                    bounds = self.scene_left.scene.bounding_box
                    self.scene_left.setup_camera(60.0, bounds, bounds.get_center())
                    self.scene_right.setup_camera(60.0, bounds, bounds.get_center())
                    self.is_first_frame = False

            gui.Application.instance.post_to_main_thread(self.window, update_ui)

    def run(self):
        # 启动主 UI 事件循环（此函数会阻塞，直到关闭窗口）
        gui.Application.instance.run()

if __name__ == '__main__':
    app = OrbbecGUIApp()
    app.run()