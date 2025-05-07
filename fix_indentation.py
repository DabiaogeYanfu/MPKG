def fix_indentation():
    with open('edc/edc_framework.py', 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_load_model = False
    
    for line in lines:
        if line.strip().startswith('def load_model(self,'):
            in_load_model = True
            # 添加正确的缩进（4个空格）
            fixed_lines.append('    ' + line)
        elif in_load_model and line.strip() and not line.startswith('    '):
            # 在load_model方法内部，添加正确的缩进（8个空格）
            fixed_lines.append('    ' + line)
        elif in_load_model and not line.strip():
            # 空行，保持原样
            fixed_lines.append(line)
        elif line.strip().startswith('def schema_definition'):
            # 到达下一个方法定义，结束load_model方法的特殊处理
            in_load_model = False
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    with open('edc/edc_framework.py', 'w') as f:
        f.writelines(fixed_lines)
    
    print("已修复edc_framework.py中的缩进问题")

if __name__ == "__main__":
    fix_indentation()