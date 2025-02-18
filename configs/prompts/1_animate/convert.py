import yaml

def add_quotes_to_yaml(file_path):
    # 读取 YAML 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 确保 'prompt' 存在并是列表
    if 'prompt' in data and isinstance(data['prompt'], list):
        data['prompt'] = [f'"{item}"' if not (item.startswith('"') and item.endswith('"')) else item for item in data['prompt']]

    # 将修改后的数据写回 YAML 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

# 使用示例
yaml_file = "700prompts.yaml"  # 替换为你的 YAML 文件路径
add_quotes_to_yaml(yaml_file)