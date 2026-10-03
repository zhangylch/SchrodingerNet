import os
import subprocess

def gpu_sel(local_size):
    """
    自动选择显存占用最小的 local_size 张卡。
    设置 CUDA_VISIBLE_DEVICES，并返回逻辑 ID 列表 [0, 1, ...]。
    """
    try:
        # 1. 使用 nvidia-smi 查询显存 (避免生成临时文件)
        # 获取显存占用，单位 MiB
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split(), encoding='utf-8')
        
        # 解析结果
        memory_usage = [int(x.strip()) for x in output.strip().split('\n') if x.strip()]
        
        # 构造 (物理ID, 显存占用) 的列表
        gpu_info = []
        for phy_id, mem in enumerate(memory_usage):
            gpu_info.append((phy_id, mem))
            
        # 检查显卡数量
        if len(gpu_info) < local_size:
            print(f"[Warning] 系统仅有 {len(gpu_info)} 张卡，但请求了 {local_size} 张。")
            # 自动降级，防止越界
            #local_size = len(gpu_info)

        # 2. 核心排序：按显存从小到大排序
        gpu_info.sort(key=lambda x: x[1])
        
        # 选取前 local_size 个最空闲的卡
        selected_gpus = gpu_info[:local_size]
        
        # 提取物理 ID
        selected_phy_ids = [str(x[0]) for x in selected_gpus]
        
        # 打印调试信息
        print(f"--- GPU Auto Select ---")
        print(f"Detected GPUs (Memory): {memory_usage}")
        print(f"Selected Physical IDs: {selected_phy_ids}")
        
        # 3. 设置环境变量
        # 这会将物理 ID 映射为逻辑 ID 0, 1, ...
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected_phy_ids)
        print(f"Exported CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # 4. 返回逻辑 ID 列表
        # JAX 初始化需要的是 [0, 1, 2...]，因为环境变量已经屏蔽了其他卡
        return list(range(local_size))

    except Exception as e:
        print(f"[Error] gpu_sel 发生错误: {e}")
        print("[Info] 回退到默认设置 (不修改 CUDA_VISIBLE_DEVICES)")
        # 兜底：返回默认序列，假设用户手动控制或 Slurm 已控制
        return list(range(local_size))
