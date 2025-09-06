import sys
import os
import torch
import inspect
import numpy as np
import gradio as gr
from importlib import import_module
import open3d as o3d
import glob
import socket
from contextlib import closing

# 模型配置库
MODEL_LIBRARY = {
    "PointConv": {
        "module": "models.pointconv",
        "class": "get_model",
        "init_args": {"num_classes": 40},
        "checkpoint": "/root/autodl-tmp/pointnet/log/pointconv_logs"
    },
    "PointNet++": {
        "module": "models.pointnet2_cls_ssg",
        "class": "get_model",
        "init_args": {"num_class": 40},
        "checkpoint": "/root/autodl-tmp/pointnet/log/pointnet2_logs"
    },
    "DGCNN": {
        "module": "models.DGCNN",
        "class": "DGCNN",
        "init_args": {"output_channels": 40},
        "checkpoint": "/root/autodl-tmp/pointnet/log/dgcnn_logs"
    }
}

# 系统初始化
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
current_model = None
model_metadata = {}

def load_model(model_name):
    """模型加载函数"""
    global current_model, model_metadata
    
    if model_name not in MODEL_LIBRARY:
        raise ValueError(f"未知模型: {model_name}")
    
    config = MODEL_LIBRARY[model_name]
    
    try:
        module = import_module(config["module"])
        model_class = getattr(module, config["class"])
        
        if inspect.isclass(model_class):
            try:
                current_model = model_class(**config["init_args"])
            except TypeError:
                current_model = model_class()
                if hasattr(current_model, 'set_num_classes'):
                    current_model.set_num_classes(40)
        else:
            current_model = model_class(**config["init_args"])
        
        if os.path.exists(config["checkpoint"]):
            checkpoint_files = sorted(
                glob.glob(f"{config['checkpoint']}/**/best_model.pth", recursive=True),
                key=os.path.getmtime, 
                reverse=True
            )
            if checkpoint_files:
                checkpoint = torch.load(checkpoint_files[0], map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                current_model.load_state_dict(state_dict, strict=False)
        
        model_metadata = {
            'name': model_name,
            'requires_normals': hasattr(current_model, 'normal_channel') and current_model.normal_channel
        }
        
        current_model.eval()
        return f"✅ {model_name} 加载成功"
    
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"

def predict(point_cloud, model_name):
    """预测函数"""
    global current_model
    
    if current_model is None or model_metadata.get('name') != model_name:
        load_model(model_name)
    
    try:
        with open(point_cloud.name, 'r') as f:
            content = f.read()
        
        content = content.replace(',', ' ')
        points = np.fromstring(content, sep=' ').astype(np.float32)
        
        if points.size % 3 != 0:
            raise ValueError("点云数据点数不是3的倍数")
        points = points.reshape(-1, 3)
        
        if model_metadata['requires_normals'] and points.shape[1] == 3:
            normals = np.zeros((points.shape[0], 3))
            normals[:, 2] = 1
            points = np.hstack([points, normals])
        elif not model_metadata['requires_normals'] and points.shape[1] > 3:
            points = points[:, :3]
        
        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] -= centroid
        scale = np.max(np.sqrt(np.sum(points[:, :3]**2, axis=1)))
        if scale > 0:
            points[:, :3] /= scale
        
        points = torch.tensor(points).float().unsqueeze(0).transpose(2, 1)
        
        with torch.no_grad():
            output = current_model(points)
            
            if isinstance(output, tuple):
                pred = output[0]
            elif isinstance(output, dict):
                pred = output.get('logits', output.get('pred_logits', output))
            else:
                pred = output
            
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            
            pred_label = torch.argmax(pred, dim=1).item()
        
        return f"模型: {model_name}\n预测类别: {pred_label}"
    
    except Exception as e:
        return f"❌ 预测失败: {str(e)}"
    
def estimate_radius(pcd, k=10):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    return avg_dist

def create_point_cloud_file(point_cloud):
    """创建三角网格PLY文件以支持3D查看器"""
    try:
        # 读取数据
        with open(point_cloud, 'r') as f:
            content = f.read().replace(',', ' ').replace('\t', ' ')
        
        # 解析数据
        points = np.fromstring(content, sep=' ').astype(np.float32)
        if points.size % 3 != 0:
            raise ValueError("点云数据点数不是3的倍数")
        points = points.reshape(-1, 3)
        
        # 创建临时目录
        temp_dir = "/tmp/gradio_pointclouds"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 设置输出文件路径（使用带面结构的 PLY 文件）
        output_filename = os.path.join(temp_dir, f"{os.path.basename(point_cloud)}_mesh.ply")

        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)  

        # 估算法线（必要）
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

        # 替换 Poisson 为 Ball Pivoting（适合稀疏点云）
        radii = [estimate_radius(pcd) * 1.5]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        # 保存三角网格（PLY）
        o3d.io.write_triangle_mesh(output_filename, mesh)

        return output_filename

    except Exception as e:
        raise gr.Error(f"创建三角网格失败: {str(e)}")

def handle_upload(file, model_name):
    """增强的上传处理函数"""
    try:
        # 预测
        prediction = predict(file, model_name)
        
        # 可视化
        model_path = create_point_cloud_file(file)
        
        # 验证文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"生成的文件不存在: {model_path}")
        
        # 验证文件大小
        if os.path.getsize(model_path) == 0:
            raise ValueError("生成的文件为空")
            
        return prediction, model_path
        
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        print(error_msg)  # 控制台日志
        return error_msg, None
    
import atexit
import shutil

# 程序退出时清理临时文件
@atexit.register
def cleanup():
    temp_dir = "/tmp/gradio_pointclouds"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# 创建界面
with gr.Blocks(title="点云分类模型切换演示") as demo:
    gr.Markdown("## 🔄 点云分类模型切换系统")
    
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=list(MODEL_LIBRARY.keys()),
            value="PointConv",
            label="选择模型架构"
        )
        upload_box = gr.File(
            label="上传点云文件",
            file_types=[".txt"],
            type="filepath"
        )
    
    with gr.Row():
        status = gr.Textbox(label="模型状态", interactive=False)
        result = gr.Textbox(label="预测结果", interactive=False)
        # 将 Image 组件更改为 Model3D 组件以实现交互式查看
        viewer = gr.Model3D(
            label="点云3D交互式可视化",
            clear_color=[0.0, 0.0, 0.0, 0.0],
            zoom_speed=1.0,  # 调整缩放速度
            height=500  # 固定高度
        )
    
    model_selector.change(
        fn=load_model,
        inputs=model_selector,
        outputs=status
    )
    
    upload_box.change(
        fn=handle_upload,
        inputs=[upload_box, model_selector],
        outputs=[result, viewer]
    )

def find_available_port(start_port=7860, end_port=8000):
    """寻找可用端口"""
    for port in range(start_port, end_port + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    raise OSError(f"No available ports between {start_port}-{end_port}")

if __name__ == "__main__":
    # 初始化默认模型
    load_model("PointConv")
    
    # 获取端口并启动
    port = find_available_port()
    print(f"⏳ 正在启动服务，请访问: http://localhost:{port}")
    
    # 禁用分析请求以避免超时错误
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )