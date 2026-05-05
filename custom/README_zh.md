# 自定义算子开发（Custom Operators）

本目录包含 **PTO 自定义算子开发示例**，展示如何从零开始实现自定义算子。

如果你刚接触 PTO 编程，建议先从基础教程入手：

- 快速入门：[docs/getting-started_zh.md](../../docs/getting-started_zh.md)
- 编程教程：[docs/coding/tutorial_zh.md](../../docs/coding/tutorial_zh.md)
- Add 算子示例：[demos/baseline/add/README_zh.md](../../demos/baseline/add/README_zh.md)

## 示例列表

- `fused_add_relu_mul/`：算子融合示例，将 Add + ReLU + Mul 融合为一个 kernel，性能提升 2-3×。

## 如何运行

每个子目录都是一个独立示例，包含各自的构建/运行说明。请从这里开始：

- [fused_add_relu_mul/README_zh.md](fused_add_relu_mul/README_zh.md)

## 开发自定义算子

参考 `fused_add_relu_mul/` 示例，按以下步骤开发：

1. 创建目录：`mkdir -p kernels/custom/my_operator`
2. 实现 kernel：`my_operator_kernel.cpp`
3. 编写测试：`main.cpp`
4. 配置构建：`CMakeLists.txt`
5. 运行验证：`./run.sh --sim`

详细开发指南请参考：

- [算子融合技术](../../docs/coding/operator-fusion_zh.md)
- [性能优化指南](../../docs/coding/opt_zh.md)
