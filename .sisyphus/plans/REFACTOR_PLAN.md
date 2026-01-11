# OpenFocus 代码重构执行计划

**创建时间**: 2026-01-11
**状态**: 待执行

## 目标

按照工程最佳实践重新组织项目结构，提升可维护性和代码组织清晰度。

## 重构任务清单

### ✅ 任务 1: 创建目录结构
- [ ] 创建 `utils/` 目录
- [ ] 创建 `core/models/` 目录
- [ ] 创建 `utils/__init__.py`
- [ ] 创建 `core/__init__.py`
- [ ] 创建 `core/models/__init__.py`

### ✅ 任务 2: 拆解并迁移 utils.py
- [ ] 创建 `utils/image_utils.py` (图像转换函数)
- [ ] 创建 `utils/ui_utils.py` (消息框函数)
- [ ] 更新 `utils/__init__.py` 统一导出
- [ ] 删除原 `utils.py`

**需要更新导入的文件**:
- [ ] `main.py`
- [ ] `controllers/source_manager.py`
- [ ] `controllers/transform_manager.py`
- [ ] `controllers/render_manager.py`
- [ ] `dialogs/about.py`
- [ ] `dialogs/settings.py`
- [ ] `dialogs/help.py`
- [ ] `dialogs/batch.py`
- [ ] `widgets/magnifier_label.py`
- [ ] `workers.py`

### ✅ 任务 3: 迁移 styles.py 到 ui/
- [ ] 重命名 `styles.py` → `ui/styles.py`

**需要更新导入的文件**:
- [ ] `main.py`
- [ ] `utils.py`
- [ ] `dialogs/about.py`
- [ ] `dialogs/settings.py`
- [ ] `dialogs/help.py`
- [ ] `dialogs/batch.py`

### ✅ 任务 4: 迁移核心模块到 core/
- [ ] 迁移 `image_loader.py` → `core/image_loader.py`
- [ ] 迁移 `Registration.py` → `core/registration.py`
- [ ] 迁移 `multi_focus_fusion.py` → `core/multi_focus_fusion.py`
- [ ] 迁移 `workers.py` → `core/workers.py`
- [ ] 创建 `core/__init__.py`

**需要更新导入的文件**:
- [ ] `main.py` (image_loader)
- [ ] `workers.py` (Registration, multi_focus_fusion)
- [ ] `controllers/source_manager.py` (image_loader)
- [ ] `core/multi_focus_fusion.py` (fusion_methods 导入)
- [ ] `core/workers.py` (更新导入路径)

### ✅ 任务 5: 迁移 network.py 到 core/models/
- [ ] 重命名 `network.py` → `core/models/stackmffv4_network.py`
- [ ] 创建 `core/models/__init__.py`

**需要更新导入的文件**:
- [ ] `fusion_methods/stackmffv4.py`

### ✅ 任务 6: 清理和验证
- [ ] 更新 `main.py` 所有导入路径
- [ ] 运行 `lsp_diagnostics` 检查错误
- [ ] 验证应用程序启动正常
- [ ] 测试核心功能

## 文件迁移详细说明

### utils/ 目录结构
```
utils/
├── __init__.py       # 统一导出所有函数
├── image_utils.py    # pixmap_to_cv2, cv2_to_pixmap
└── ui_utils.py       # show_message_box 系列函数
```

### core/ 目录结构
```
core/
├── __init__.py
├── image_loader.py           # ImageStackLoader
├── registration.py           # ImageRegistration
├── multi_focus_fusion.py     # MultiFocusFusion
├── workers.py                # RenderWorker
└── models/
    ├── __init__.py
    └── stackmffv4_network.py # StackMFF_V4, LV_UNet
```

## 向后兼容性

本次重构**不提供**向后兼容导入路径，直接更新所有导入。

## 风险点

1. **workers.py 导入变更**: 影响配准和融合功能
2. **fusion_methods/stackmffv4.py 导入**: 影响神经网络融合
3. **大量文件导入更新**: 需要全面测试

## 测试验证清单

- [ ] 应用程序启动
- [ ] 图像加载功能
- [ ] 图像配准 (Homography, ECC)
- [ ] 融合算法 (5种方法)
- [ ] StackMFF-V4 融合
- [ ] UI 对话框
- [ ] 国际化 (中英文切换)
- [ ] 批处理功能

## 回滚计划

如果重构出现问题，执行以下命令回滚:
```bash
git checkout HEAD -- \
  utils.py styles.py \
  image_loader.py Registration.py \
  multi_focus_fusion.py workers.py \
  network.py
```
