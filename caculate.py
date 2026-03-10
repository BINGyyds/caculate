import pandas as pd
import numpy as np
import pinocchio as pin
import os

# ================= 配置区域 =================
EXCEL_FILE_PATH = 'traj_ling.xlsx'  # 替换为你的真实xlsx文件名
URDF_BASE_DIR = 'urdf/'        # 确保这个路径相对当前运行目录是正确的

# 【非常重要】: 替换为你大臂 URDF 中末端连杆 (Link) 或 Frame 的真实名字！
# 可以打开 big_jxb_standalone.urdf 搜一下最后一个 link 的 name
BIG_ARM_EE_NAME = 'biglink7' 

urdf_paths = {
    'big_arm': {
        'standalone': os.path.join(URDF_BASE_DIR, 'big_jxb_standalone.urdf'),
        'with_medium': os.path.join(URDF_BASE_DIR, 'big_jxb_with_medium_jxb.urdf'),
        'with_medium_and_load': os.path.join(URDF_BASE_DIR, 'big_jxb_with_medium_and_load.urdf')
    },
    'medium_arm': {
        'standalone': os.path.join(URDF_BASE_DIR, 'medium_jxb_standalone.urdf'),
        'with_load': os.path.join(URDF_BASE_DIR, 'medium_jxb_with_load.urdf')
    }
}

# ================= 辅助函数 =================
def load_robot_model(urdf_path):
    """加载Pinocchio机器人模型"""
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"找不到URDF文件，请检查路径: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

def calculate_mechanical_power(model, data, q, v, a):
    """通过逆动力学(RNEA)计算力矩，并计算当前机械功率"""
    # 确保输入的 q, v, a 维度和 URDF 模型定义的自由度匹配
    if len(q) != model.nq or len(v) != model.nv or len(a) != model.nv:
        # 如果不匹配，通常是因为 URDF 里写死了 base_link 或者有多余的 dummy joint
        # 这里做截断或补零以防报错 (根据实际情况可能需要修改)
        q = np.resize(q, model.nq)
        v = np.resize(v, model.nv)
        a = np.resize(a, model.nv)
        
    tau = pin.rnea(model, data, q, v, a)
    # 计算机械功率 P = sum(|tau_i * v_i|)
    power = np.sum(np.abs(tau * v))
    return power

# ================= 主程序 =================
def main():
    print("正在加载 Excel 文件，这需要占用一定内存并消耗几分钟时间...")
    all_sheets = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
    print(f"成功加载，共发现 {len(all_sheets)} 个 Sheet 工步。")
    
    total_time = 0.0
    total_distance = 0.0
    total_energy = 0.0
    
    for sheet_name, df in all_sheets.items():
        print(f"正在处理 Sheet: {sheet_name} (共 {len(df)} 行数据)")
        
        # 1. 提取时间
        time_array = df['time'].values
        if len(time_array) < 2:
            print(f"  -> Sheet {sheet_name} 数据行数不足，跳过。")
            continue
            
        step_time = time_array[-1] - time_array[0]
        total_time += step_time
        dt_array = np.diff(time_array)
            
        # 提取当前 sheet 的配置状态
        big_load_type = df['big_load_type'].iloc[0]
        medium_load_type = df['medium_load_type'].iloc[0]
        
        # 加载对应模型
        try:
            big_model, big_data = load_robot_model(urdf_paths['big_arm'][big_load_type])
            med_model, med_data = load_robot_model(urdf_paths['medium_arm'][medium_load_type])
        except Exception as e:
            print(f"  -> 加载 URDF 失败: {e}，跳过该 Sheet。")
            continue
        
        # 获取大臂末端 ID —— 这里修正了 existFrame 的调用方式！
        if big_model.existFrame(BIG_ARM_EE_NAME):
            ee_id = big_model.getFrameId(BIG_ARM_EE_NAME)
        else:
            print(f"  -> 警告: URDF中找不到名为 '{BIG_ARM_EE_NAME}' 的Frame，将默认使用最后一个关节作为末端。")
            # 如果名字填错了，容错处理：默认取最后一个 frame
            ee_id = len(big_model.frames) - 1

        # 提取运动学数据 7 轴
        big_q = df[[f'bigjoint{i}_pos' for i in range(1, 8)]].values
        big_v = df[[f'bigjoint{i}_vel' for i in range(1, 8)]].values
        big_a = df[[f'bigjoint{i}_accel' for i in range(1, 8)]].values
        
        med_q = df[[f'middlejoint{i}_pos' for i in range(1, 8)]].values
        med_v = df[[f'middlejoint{i}_vel' for i in range(1, 8)]].values
        med_a = df[[f'middlejoint{i}_accel' for i in range(1, 8)]].values

        prev_ee_pos = None
        
        # 遍历每行进行计算
        for i in range(len(df)):
            # =========== 计算大臂末端移动距离 ===========
            # 正运动学 (需要把 7 个关节角转化为 numpy 连续数组)
            q_current = np.ascontiguousarray(big_q[i])
            
            # 如果 Excel 数据自由度(7)和 URDF 不一致，做强制截断适配
            if len(q_current) != big_model.nq:
                q_current = np.resize(q_current, big_model.nq)
                
            pin.forwardKinematics(big_model, big_data, q_current)
            pin.updateFramePlacements(big_model, big_data)
            
            # 获取三维坐标 [x, y, z]
            current_ee_pos = big_data.oMf[ee_id].translation.copy()
            
            if prev_ee_pos is not None:
                dist = np.linalg.norm(current_ee_pos - prev_ee_pos)
                total_distance += dist
            prev_ee_pos = current_ee_pos
            
            # =========== 计算消耗的总能量 ===========
            if i < len(dt_array):
                dt = dt_array[i]
                if dt <= 0: continue # 防止时间倒退的异常数据
                
                # 计算功率
                big_power = calculate_mechanical_power(
                    big_model, big_data, 
                    np.ascontiguousarray(big_q[i]), 
                    np.ascontiguousarray(big_v[i]), 
                    np.ascontiguousarray(big_a[i])
                )
                med_power = calculate_mechanical_power(
                    med_model, med_data, 
                    np.ascontiguousarray(med_q[i]), 
                    np.ascontiguousarray(med_v[i]), 
                    np.ascontiguousarray(med_a[i])
                )
                
                # 能量累加
                total_energy += (big_power + med_power) * dt

    print("\n" + "="*45)
    print("================ 最终计算结果 ================")
    print(f"全流程总时间: {total_time:.2f} 秒")
    print(f"大臂末端总移动距离: {total_distance:.4f} 米")
    print(f"全流程总消耗机械能: {total_energy/1000:.2f} 千焦 (KJ)")
    # print(f"全流程总时间: {total_time/12:.2f} 秒")
    # print(f"大臂末端总移动距离: {total_distance/12:.4f} 米")
    # print(f"全流程总消耗机械能: {total_energy/1000/12:.2f} 千焦 (KJ)")
    print("="*45)

if __name__ == '__main__':
    main()