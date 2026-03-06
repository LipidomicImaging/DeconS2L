import torch
import os

class Cfg:
    # --- 硬件配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    
    # --- 数据索引 ---
    # ce=768; le=3; ri=5
    # ce=710; le=2; ri=5
    # ce=808;le=3;ri=0;
    # ce=758;le=3;ri=0;
    ce=772;le=3;ri=0
    # ce=786;le=3;ri=0
    # ce=748;le=5;ri=0
    # ce=787;le=5;ri=0;Windows=(786, 794)
    # ce=770;le=4;ri=3;Windows=(765, 789)
    # ce=700;le=0;ri=0;Windows=(750, 890)
    # ce=756;le=4;ri=3;Windows=(756, 779)
    raw_spectrum_path = "spectrum_hun.npy"
    raw_data_path     = "data_hun.npy"
    mz_path           = 'shared_mz_hun.npy'
    meta_path='processed_isomer_data_filter_slim_hun.npy'
    # raw_spectrum_path = "spectrum_%do%d_%d.npy"%(le,ri,ce)
    # raw_data_path     = "data_%do%d_%d.npy"%(le,ri,ce)
    # mz_path           = 'shared_mz1_%do%d_%d.npy'%(le,ri,ce)
    # meta_path='processed_isomer_data_filter_%do%d_%d_slim.npy'%(le,ri,ce)
    
    train_data_dir = "data/train"
    processed_A_path = os.path.join(train_data_dir, "A_library.npy")
    processed_B_path = os.path.join(train_data_dir, "B_cube.npy")
    
    out_dir = "results_auto_weighted" 
    
    # --- 训练超参数 ---
    # 单样本优化通常需要更多步数来收敛
    n_epochs = 155000     
    batch_size = 1      # 单样本
    lr_net = 1e-4       # 网络的学习率
    lr_weight = 5e-4    # 损失权重的学习率 (通常稍微大一点)
    
    grad_clip_norm = 1.0 
    
    # --- 架构参数 ---
    K_layers = 12       # 展开深度
    full_image_shape = (200, 90) # 根据实际数据修改
    
    # --- 策略开关 ---
    warmup_epochs = 500     # 谱库冻结期
    calib_clamp_min = 0.5
    calib_clamp_max = 1.5
    
    # --- 可视化 ---
    vis_freq = 100
    save_freq = 500
