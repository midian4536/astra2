from pyorbbecsdk import Pipeline, OBSensorType

def list_device_methods():
    """
    列出 pyorbbecsdk.Device 类的所有可用方法
    """
    pipeline = Pipeline()
    device = pipeline.get_device()
    
    if device is None:
        print("无法获取设备对象")
        return
    
    print("=" * 60)
    print("pyorbbecsdk.Device 类的所有可用方法:")
    print("=" * 60)
    
    methods = [method for method in dir(device) if not method.startswith('_')]
    
    for i, method in enumerate(methods, 1):
        print(f"{i:2d}. {method}")
    
    print("=" * 60)
    print(f"总计: {len(methods)} 个方法")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("从代码中实际使用的方法:")
    print("=" * 60)
    print("1. get_camera_calibration(sensor_type) - 获取相机校准数据")
    print("2. get_extrinsics(from_sensor, to_sensor) - 获取外参（旋转和平移）")
    print("=" * 60)

if __name__ == "__main__":
    list_device_methods()
