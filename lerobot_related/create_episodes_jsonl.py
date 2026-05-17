import pandas as pd
import os
import json

# 1. 定义基础路径
base_path = '/home/qwe/isaacsim_piper/isaacsim_piper/isaacsim_piper/my_data/'
data_dir = os.path.join(base_path, 'data/chunk_000')
meta_dir = os.path.join(base_path, 'meta')

# 2. 获取所有的 parquet 文件并排序
if not os.path.exists(data_dir):
    print(f"错误: 找不到数据目录 {data_dir}")
    exit()

onlyfiles = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
onlyfiles.sort()

# 3. 预先读取 tasks.jsonl 内容
tasks = []
tasks_path = os.path.join(meta_dir, 'tasks.jsonl')
if os.path.exists(tasks_path):
    with open(tasks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
else:
    print(f"警告: 找不到 {tasks_path}")

jsonl_data = []

# 4. 遍历处理每个 parquet 文件
for file in onlyfiles:
    file_path = os.path.join(data_dir, file)
    
    # 读取 parquet
    df = pd.read_parquet(file_path)
    
    # 打印调试信息
    print(f"处理文件: {file} | 第一行索引: {df['index'].iloc[0]} | 最后一行索引: {df['index'].iloc[-1]}")
    
    # 构建当前 episode 的字典
    episode_dic = {}
    episode_dic['episode_index'] = int(df['episode_index'].iloc[0])
    
    # 关联 task 信息 (假设 tasks[0] 对应所有数据，或者根据索引匹配)
    if tasks:
        # 这里默认取第一个 task，如果你的任务和文件是一一对应的，可以用 tasks[i]
        episode_dic['tasks'] = [tasks[0]['task']]
        episode_dic['task_index'] = [tasks[0]['task_index']]
    
    episode_dic['length'] = len(df)
    jsonl_data.append(episode_dic)

# 5. 保存到 my_data/meta 文件夹下
output_file = os.path.join(meta_dir, 'episodes.jsonl')

with open(output_file, 'w', encoding='utf-8') as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + "\n")

print(f"\n成功！文件已保存至: {output_file}")