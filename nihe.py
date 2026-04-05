import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 设置 Matplotlib 支持中文显示（如果需要中文标题，否则请注销这行）
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False  

def calculate_calibration_models_with_plots(csv_file_path):
    # 1. 加载提取好的坐标数据
    df = pd.read_csv(csv_file_path)

    # 创建大图，显著加宽 figsize=(24, 6) 防止挤压重叠
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle('相机坐标校准前后对比 (散点图)', fontsize=18, fontweight='bold')

    # ================= 1. X轴拟合与绘图 =================
    df_x = df.dropna(subset=['Target_X', '3D_X', '3D_Z']).copy()
    model_x = LinearRegression()
    model_x.fit(df_x[['3D_X', '3D_Z']], df_x['Target_X'])
    df_x['X_calibrated'] = model_x.predict(df_x[['3D_X', '3D_Z']])
    
    mae_x_before = np.mean(np.abs(df_x['3D_X'] - df_x['Target_X']))
    mae_x_after = np.mean(np.abs(df_x['X_calibrated'] - df_x['Target_X']))

    ax = axs[0] # 第1张图 X轴
    min_val = min(df_x['Target_X'].min(), df_x['3D_X'].min())
    max_val = max(df_x['Target_X'].max(), df_x['3D_X'].max())
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax.scatter(df_x['Target_X'], df_x['3D_X'], color='blue', alpha=0.6, s=50, label=f'Before (MAE={mae_x_before:.1f})')
    ax.scatter(df_x['Target_X'], df_x['X_calibrated'], color='green', alpha=0.6, s=50, label=f'After (MAE={mae_x_after:.1f})')
    
    ax.set_title('X-Axis Calibration', fontsize=14)
    ax.set_xlabel('True Target X (mm)', fontsize=12)
    ax.set_ylabel('Measured/Calibrated X (mm)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    # ================= 2. Y轴拟合与绘图 =================
    df_y = df.dropna(subset=['Target_Y', '3D_Y', '3D_Z']).copy()
    model_y = LinearRegression()
    model_y.fit(df_y[['3D_Y', '3D_Z']], df_y['Target_Y'])
    df_y['Y_calibrated'] = model_y.predict(df_y[['3D_Y', '3D_Z']])
    
    mae_y_before = np.mean(np.abs(df_y['3D_Y'] - df_y['Target_Y']))
    mae_y_after = np.mean(np.abs(df_y['Y_calibrated'] - df_y['Target_Y']))

    ax = axs[1] # 第2张图 Y轴
    min_val = min(df_y['Target_Y'].min(), df_y['3D_Y'].min())
    max_val = max(df_y['Target_Y'].max(), df_y['3D_Y'].max())
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax.scatter(df_y['Target_Y'], df_y['3D_Y'], color='blue', alpha=0.6, s=50, label=f'Before (MAE={mae_y_before:.1f})')
    ax.scatter(df_y['Target_Y'], df_y['Y_calibrated'], color='green', alpha=0.6, s=50, label=f'After (MAE={mae_y_after:.1f})')
    
    ax.set_title('Y-Axis Calibration', fontsize=14)
    ax.set_xlabel('True Target Y (mm)', fontsize=12)
    ax.set_ylabel('Measured/Calibrated Y (mm)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    # ================= 3. Z轴拟合与绘图 =================
    df_z = df.dropna(subset=['Target_Z', '3D_Z']).copy()
    model_z = LinearRegression()
    model_z.fit(df_z[['3D_Z']], df_z['Target_Z'])
    df_z['Z_calibrated'] = model_z.predict(df_z[['3D_Z']])
    
    mae_z_before = np.mean(np.abs(df_z['3D_Z'] - df_z['Target_Z']))
    mae_z_after = np.mean(np.abs(df_z['Z_calibrated'] - df_z['Target_Z']))

    ax = axs[2] # 第3张图 Z轴
    min_val = min(df_z['Target_Z'].min(), df_z['3D_Z'].min())
    max_val = max(df_z['Target_Z'].max(), df_z['3D_Z'].max())
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax.scatter(df_z['Target_Z'], df_z['3D_Z'], color='blue', alpha=0.6, s=50, label=f'Before (MAE={mae_z_before:.1f})')
    ax.scatter(df_z['Target_Z'], df_z['Z_calibrated'], color='green', alpha=0.6, s=50, label=f'After (MAE={mae_z_after:.1f})')
    
    ax.set_title('Z-Axis Calibration', fontsize=14)
    ax.set_xlabel('True Target Z (mm)', fontsize=12)
    ax.set_ylabel('Measured/Calibrated Z (mm)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 引入 w_pad 明确拉开子图之间的水平间距，防止坐标轴重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=4.0)
    
    # 显示与保存
    # plt.savefig('calibration_plot.png', dpi=300, bbox_inches='tight') # 如果需要保存成高清图片，可以取消注释此行
    plt.show()

    return model_x, model_y, model_z

if __name__ == "__main__":
    model_x, model_y, model_z = calculate_calibration_models_with_plots('extracted_coordinates.csv')