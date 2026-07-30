import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ==========================================
# ⚠️ 請修改這裡：你的 JSON 檔案路徑
# ==========================================
json_path = 'data/raw/P001_Sit.json' 
# 例如：'C:/Users/YourName/Desktop/project/data/raw/P001_Sit.json'
# ==========================================

# 定義 MediaPipe 33 個點之間的連線關係 (骨架)
# 參考 MediaPipe 官方文檔，定義哪些點需要連成線
POSE_CONNECTIONS = [
    # 軀幹
    (11, 12), (12, 24), (24, 23), (23, 11),
    # 右臂
    (12, 14), (14, 16),
    # 左臂
    (11, 13), (13, 15),
    # 右腿 (我們模型關注的核心)
    (24, 26), (26, 28), (28, 32), (32, 30), (30, 28),
    # 左腿 (我們模型關注的核心)
    (23, 25), (25, 27), (27, 31), (31, 29), (29, 27)
]

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 提取所有幀的座標 (N_frames, 33, 3)
    all_frames_coords = []
    for frame in data:
        current_frame_coords = []
        for landmark in frame:
            # MediaPipe 的 z 通常很小，為了視覺化明顯，我們保持原樣
            current_frame_coords.append([landmark['x'], landmark['y'], landmark['z']])
        all_frames_coords.append(current_frame_coords)
    
    return np.array(all_frames_coords)

def preprocess_for_vis(coords):
    """
    【終極修正版】徹底解決「平躺做仰臥起坐」問題。
    將 MediaPipe 的坐姿動作正確映射到標準 3D 繪圖空間（Z軸向上）。
    """
    processed = coords.copy()
    
    # 提取原始 MediaPipe 座標
    # 原本的 (N, 33, 0)=X, (N, 33, 1)=Y, (N, 33, 2)=Z
    orig_x = processed[:, :, 0]
    orig_y = processed[:, :, 1]
    orig_z = processed[:, :, 2]

    # 🏆 軸向重新映射魔法 (Axis Mapping Magic) 🏆
    # 我們要建立一個新的座標矩陣
    new_coords = np.zeros_like(processed)
    
    # 1. 繪圖 X = 原始 X (保持左右方向)
    new_coords[:, :, 0] = orig_x
    
    # 2. 繪圖 Y (地面深度) = 原始 Z (將原本的深度變為地面的縱深)
    new_coords[:, :, 1] = orig_z
    
    # 3. 繪圖 Z (垂直高度) = -原始 Y (最關鍵！將影片的上下 Y 變成標準高度 Z，-號讓頭朝上)
    new_coords[:, :, 2] = -orig_y
    
    # --- 視覺優化：讓屁股（骨盆）回到原點，且站在畫面上 ---
    # 計算第一幀骨盆的中點 (第 23, 24 點) 在新座標系下的位置
    mid_hip_start = (new_coords[0, 23, :] + new_coords[0, 24, :]) / 2
    
    # 全體扣除骨盆起始點，讓人核心居中
    zero_centered = new_coords - mid_hip_start
    
    return zero_centered    

# 1. 載入與預處理資料
try:
    print(f"正在載入資料：{json_path} ...")
    raw_coords = load_json_data(json_path)
    # 處理視覺化用的座標 (N, 33, 3)
    coords = preprocess_for_vis(raw_coords)
    num_frames = coords.shape[0]
    print(f"載入成功！總共 {num_frames} 幀。正在啟動 3D 視窗...")
except FileNotFoundError:
    print(f"❌ 錯誤：找不到檔案，請檢查路徑是否正確：{json_path}")
    exit()

# 2. 設定 Matplotlib 3D 繪圖視窗
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 設定視窗標題與背景（全白，方便截圖）
fig.canvas.manager.set_window_title('肌少症 AI - 純淨 3D 骨架視覺化')
ax.set_facecolor('white') # 雖然設定了，但主要是要把下面的 axes 關掉

# 🏆 關鍵：關掉座標軸、網格、背景，達成「純淨」效果
ax.set_axis_off() 

# 設定初始視角 (這是一個很棒的 45 度立體視角)
ax.view_init(elev=20, azim=45)

# 初始化繪圖物件 (點和線)
scat = ax.scatter([], [], [], c='blue', s=20) # 關節點（藍色）
lines = [ax.plot([], [], [], c='red', linewidth=2)[0] for _ in POSE_CONNECTIONS] # 骨骼（紅色）

# 3. 定義動畫更新函數
def update(frame_idx):
    current_points = coords[frame_idx] # (33, 3)
    
    # 更新關節點
    # Matplotlib 3D scatter 更新比較特殊，需要用 _offsets3d
    scat._offsets3d = (current_points[:, 0], current_points[:, 1], current_points[:, 2])
    
    # 更新骨骼連線
    for line, connection in zip(lines, POSE_CONNECTIONS):
        start_idx, end_idx = connection
        xs = [current_points[start_idx, 0], current_points[end_idx, 0]]
        ys = [current_points[start_idx, 1], current_points[end_idx, 1]]
        zs = [current_points[start_idx, 2], current_points[end_idx, 2]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
    
    return [scat] + lines

# 4. 啟動動畫
# interval=33 對應 30 FPS，讓動作速度看起來正常
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=33, blit=False, repeat=True)

# 顯示視窗 (在此視窗中你可以用滑鼠旋轉找到完美的截圖角度)
print("\n👉 提示：你可以用滑鼠在彈出的視窗中拖曳旋轉，找到最完美的角度後截圖放進 PPT！")
plt.show()