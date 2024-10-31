#陈子豪2024.1.7 论文最终版本
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO
import cv2


model = YOLO('C:/Users/Administrator/Desktop/ultralytics-main/runs/detect/train4/weights/best.pt')
def wait_for_frames(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    img_color = np.asanyarray(color_frame.get_data())
    img_depth = np.asanyarray(depth_frame.get_data())
    return color_frame, depth_frame, img_color, img_depth


def Pointcloud_depth_main(pc, box, depth_frame):
    depth_pixel = [int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)]
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    vtx = np.reshape(vtx, (480, 640, -1))
    x, y = depth_pixel
    camera_coordinate = vtx[y][x][0]
    Pointcloud_depth = camera_coordinate[2]
    return Pointcloud_depth

# 定义滤波器
def aukf(f, x, P, h, z, Q, R):
    n = len(x)  # 状态维度
    X = np.concatenate((np.array([x]).T, x + np.linalg.cholesky(P).T, x - np.linalg.cholesky(P).T), axis=1)
    X = f(X)
    x = np.mean(X, axis=1)
    P_prev = P
    P = np.cov(X) + Q

    Y = h(X)
    y = np.mean(Y, axis=1)
    Pyy = np.cov(Y) + R
    Pxy = np.cov(X, Y)[0:n, n:]
    K = np.dot(Pxy, np.linalg.inv(Pyy))

    x = x + np.dot(K, z - y)

    P = P - np.dot(np.dot(K, Pyy), K.T)

    Q = Q * np.maximum(1, np.abs(np.diag(P - P_prev)) / np.diag(Q))
    R = R * np.maximum(1, np.abs(z - h(x)) / np.diag(R))

    return x, P, Q, R


# RealSense深度相机初始化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

profile = pipeline.start(config)

pc = rs.pointcloud()
points = rs.points()

# 互补滤波器参数
alpha = 0.65  # 互补滤波系数，加速度
filtered_acceleration = np.zeros(3)
filtered_angular_velocity = np.zeros(3)


# 获取深度和IMU数据
num_data_points = 100     #指定了要收集的数据点数量

#初始化数据
depth_data = np.zeros(num_data_points)
fused_depth_data = np.zeros(num_data_points)

# 初始化状态 x
initial_depth = 0.0
initial_acceleration = np.zeros(3)  # 初始化为零向量  初始加速度
initial_angular_velocity = np.zeros(3)  # 初始化为零向量   初始角速度
x = np.concatenate(([initial_depth], initial_acceleration, initial_angular_velocity))

# 初始化滤波器
initial_covariance = np.eye(len(x)) * 0.1    #初始化状态协方差矩阵
initial_process_noise_std = 0.1   #过程噪声的初始标准差
initial_measurement_noise_std = 0.1  #测量噪声的初始标准差
Q = np.eye(len(x)) * initial_process_noise_std**2   #根据过程噪声标准差计算过程噪声协方差矩阵
R = np.eye(1) * initial_measurement_noise_std**2    #根据测量噪声标准差计算测量噪声协方差矩阵。
P = initial_covariance

plt.ion()
# 获取深度和IMU数据并进行滤波
depth_data_value = 0.0
for i in range(num_data_points):
    # 读取RealSense D435i深度和IMU数据
    frames = pipeline.wait_for_frames()
    color_frame, depth_frame, img_color, img_depth = wait_for_frames(pipeline)
    results = model(img_color)
    boxs = results[0].boxes.xyxy  # 提取yolo左上角 和右下角代码 并且转化为numpy
    accel_frame = frames.first_or_default(rs.stream.accel)
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    # 获取深度信息
    for box in boxs:
        depth_data_value = Pointcloud_depth_main(pc, box, depth_frame)


    # 获取IMU数据
    imu_acceleration_value = np.array([accel_frame.as_motion_frame().get_motion_data().x,
                                       accel_frame.as_motion_frame().get_motion_data().y,
                                       accel_frame.as_motion_frame().get_motion_data().z])
    imu_angular_velocity_value = np.array([gyro_frame.as_motion_frame().get_motion_data().x,
                                           gyro_frame.as_motion_frame().get_motion_data().y,
                                           gyro_frame.as_motion_frame().get_motion_data().z])


    # 互补滤波
    filtered_acceleration = alpha * filtered_acceleration + (1 - alpha) * imu_acceleration_value
    filtered_angular_velocity = alpha * filtered_angular_velocity + (1 - alpha) * imu_angular_velocity_value

    # 第一次使用滤波器时，初始化状态 x
    if i == 0:
        x[0] = depth_data_value

    # 更新状态方程中的IMU数据
    x[1:4] = filtered_acceleration
    x[4:] = filtered_angular_velocity

    # 进行滤波
    x, P, Q, R = aukf(lambda x: x, x, P, lambda x: x, np.array([depth_data_value]), Q, R)

    # 获取融合后的深度信息
    fused_depth = x[0]


    # 记录数据
    depth_data[i] = depth_data_value
    fused_depth_data[i] = fused_depth

    # 高斯均值滤波
    sigma = 3  # 选择合适的标准差
    smoothed_fused_depth_data = gaussian_filter1d(fused_depth_data[:i + 1], sigma=sigma)
    print(smoothed_fused_depth_data[0])

    # 实时绘制图像以及更新imu融合前后数据
    plt.clf()
    plt.plot(range(i + 1), depth_data[:i + 1], label='RealSense Depth')
    plt.plot(range(i + 1), fused_depth_data[:i + 1], label='Fused Depth (Raw)')
    plt.plot(range(i + 1), smoothed_fused_depth_data, label=f'Fused Depth (Smoothed, σ={sigma})')
    plt.xlabel('Time Steps')
    plt.ylabel('Depth')
    plt.legend()
    plt.pause(0.01)  # Adjust the pause duration as needed



# 停止RealSense深度相机
pipeline.stop()

# 关闭交互模式
plt.ioff()

# 显示最终结果
plt.show()