---
description: 创建新的课程文件
---

# 创建新课程文件工作流

使用此工作流创建新的课程文件。

## 步骤

1. 确定课程所属的阶段目录 (`src/phase-X-xxx/`)

2. 查看该目录下已有的文件，确定新课程的序号

```bash
ls src/{phase-目录}/
```

3. 参考 `.agent/skills/create-lesson/SKILL.md` 中的模板创建新课程文件

4. 使用课程模板创建文件，包含：

   - 文档字符串（学习目标、核心概念、前置知识）
   - 理论介绍部分
   - 基础实现部分
   - 进阶应用部分
   - 练习与思考部分
   - 主函数

5. 运行课程文件验证无错误

```bash
source venv/bin/activate && python {新课程文件路径}
```

6. 如果该目录有 README.md，更新文件列表
