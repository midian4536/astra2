import sys, math, time, numpy as np, rtde_control, rtde_receive, csv, os
from loguru import logger

ROBOT_IP = "192.168.11.5"
START_POSE = [0, -0.4, 0.3, math.pi, 0, 0]

CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), "1.csv")

def read_target_pose_from_csv():
    points = []
    with open(CSV_FILE_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                x_mm = float(row[1])
                y_mm = float(row[2])
                z_mm = float(row[3])
                points.append([x_mm, y_mm, z_mm])
    
    if len(points) < 4:
        raise ValueError(f"CSV文件中需要至少4个点，当前只有{len(points)}个点")
    
    points_array = np.array(points)
    avg_x_mm = np.mean(points_array[:, 0])
    avg_y_mm = np.mean(points_array[:, 1])
    avg_z_mm = np.mean(points_array[:, 2])
    
    avg_x_m = avg_x_mm / 1000.0
    avg_y_m = avg_y_mm / 1000.0
    avg_z_m = avg_z_mm / 1000.0
    
    logger.info(f"B1-B4平均坐标(mm): X={avg_x_mm:.2f}, Y={avg_y_mm:.2f}, Z={avg_z_mm:.2f}")
    logger.info(f"B1-B4平均坐标(m): X={avg_x_m:.4f}, Y={avg_y_m:.4f}, Z={avg_z_m:.4f}")
    
    return [avg_x_m, avg_y_m, avg_z_m, math.pi/2, -math.pi/4, -math.pi/2]

TARGET_POSE = read_target_pose_from_csv()
offset = [0, 0, -0.2, 0, 0, 0]         
logger.remove()
logger.add(
    sys.stdout,
    format="{time} | {level} | {message}",
    level="INFO",
    enqueue=True,
)

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c.zeroFtSensor()

MIDDLE_POSE = rtde_c.poseTrans(TARGET_POSE, offset)
logger.info(f"MIDDLE_POSE: {MIDDLE_POSE}")

start_pose_joint = rtde_c.getInverseKinematics(START_POSE, rtde_r.getActualQ())
middle_pose_joint = rtde_c.getInverseKinematics(MIDDLE_POSE, rtde_r.getActualQ())
target_pose_joint = rtde_c.getInverseKinematics(TARGET_POSE, rtde_r.getActualQ())

rtde_c.moveJ(start_pose_joint, 0.2, 0.1)
logger.info("已移动到 START_POSE")
time.sleep(1)

user_input = input("是否继续移动到 MIDDLE_POSE? (输入 y 继续): ")
if user_input.lower() != 'y':
    logger.info("用户取消移动，程序结束")
    rtde_c.stopScript()
    sys.exit(0)

rtde_c.moveJ(middle_pose_joint, 0.2, 0.1)
logger.info("已移动到 MIDDLE_POSE")
time.sleep(1)

user_input = input("是否继续移动到 TARGET_POSE? (输入 y 继续): ")
if user_input.lower() != 'y':
    logger.info("用户取消移动，程序结束")
    rtde_c.stopScript()
    sys.exit(0)

rtde_c.moveL(TARGET_POSE, 0.2, 0.1)
logger.info("已移动到 TARGET_POSE")
time.sleep(1)

rtde_c.stopScript()
logger.info("Done.")