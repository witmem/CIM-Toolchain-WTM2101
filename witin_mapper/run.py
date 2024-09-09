import os
import subprocess
import shutil

# 定义要遍历的文件夹路径
folder_path = './tests/onnx/witin/wtm2101/precision/'
# 保存文件路径
output_path = './output/'

# 定义不想执行的文件名列表
exclude_files = ['test_auto_optimizer_dccrn_skip_connect.py', 
'test_auto_optimizer_tc_resnet_cnn.py', 'test_auto_optimizer_lstm_manual_output_repeat.py',
'test_forward_auto_opti_denoise.py', "test_forward_tc_resnet_cnn.py", "test_forward_tc_resnet_cnn_fake.py"]

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    # 检查文件是否是Python文件，并且不在排除列表中
    if filename.endswith('.py') and filename not in exclude_files:
        # 构建要执行的命令
        command = ['python3', os.path.join(folder_path, filename)]
        
        # 执行命令
        subprocess.run(command)

# for filename in os.listdir(output_path):
#     file_path = os.path.join(output_path, filename)
#     try:
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print(f"Failed to delete {file_path}: {e}")