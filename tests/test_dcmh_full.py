"""
DCMH 模型完整测试与分析报告

生成详细的测试结果、统计图表和实验报告。
"""

import torch
import json
import time
from datetime import datetime
from pathlib import Path

# 导入 DCMH 模块
from core.hashing.dcmh_basic import DCMHBasicModule
from core.hashing.dcmh_image import DCMHImageModule, build_dcmh_image_model
from core.hashing.dcmh_text import DCMHTextModule, build_dcmh_text_model
from core.hashing.dcmh_model import DCMHModel, DCMHWithQuantization, build_dcmh_model
from training.dcmh_loss import DCMHCombinedLoss, DCMHQuantizationLoss, DCMHInfoNCELoss
from config.model_config import DCMH_CONFIG, DCMH_BIT_CONFIGS


def test_module_imports():
    """测试所有模块导入。"""
    print("=" * 60)
    print("1. 模块导入测试")
    print("=" * 60)

    results = []

    # 基础模块
    try:
        from core.hashing.dcmh_basic import DCMHBasicModule
        results.append({"module": "DCMHBasicModule", "status": "PASS"})
        print("✓ DCMHBasicModule 导入成功")
    except Exception as e:
        results.append({"module": "DCMHBasicModule", "status": "FAIL", "error": str(e)})
        print(f"✗ DCMHBasicModule 导入失败：{e}")

    # 图像模块
    try:
        from core.hashing.dcmh_image import DCMHImageModule
        results.append({"module": "DCMHImageModule", "status": "PASS"})
        print("✓ DCMHImageModule 导入成功")
    except Exception as e:
        results.append({"module": "DCMHImageModule", "status": "FAIL", "error": str(e)})
        print(f"✗ DCMHImageModule 导入失败：{e}")

    # 文本模块
    try:
        from core.hashing.dcmh_text import DCMHTextModule
        results.append({"module": "DCMHTextModule", "status": "PASS"})
        print("✓ DCMHTextModule 导入成功")
    except Exception as e:
        results.append({"module": "DCMHTextModule", "status": "FAIL", "error": str(e)})
        print(f"✗ DCMHTextModule 导入失败：{e}")

    # 双流模型
    try:
        from core.hashing.dcmh_model import DCMHModel
        results.append({"module": "DCMHModel", "status": "PASS"})
        print("✓ DCMHModel 导入成功")
    except Exception as e:
        results.append({"module": "DCMHModel", "status": "FAIL", "error": str(e)})
        print(f"✗ DCMHModel 导入失败：{e}")

    # 损失函数
    try:
        from training.dcmh_loss import DCMHCombinedLoss
        results.append({"module": "DCMHCombinedLoss", "status": "PASS"})
        print("✓ DCMHCombinedLoss 导入成功")
    except Exception as e:
        results.append({"module": "DCMHCombinedLoss", "status": "FAIL", "error": str(e)})
        print(f"✗ DCMHCombinedLoss 导入失败：{e}")

    return results


def test_image_module():
    """测试图像模块。"""
    print("\n" + "=" * 60)
    print("2. 图像模块测试")
    print("=" * 60)

    results = {}

    # 创建模型
    bit = 64
    model = DCMHImageModule(bit=bit)

    # 测试不同 batch size
    batch_sizes = [1, 2, 4, 8]
    inference_times = []

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        # 推理时间测试
        start = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        elapsed = time.time() - start
        inference_times.append(elapsed)

        results[f"batch_{batch_size}"] = {
            "input_shape": list(dummy_input.shape),
            "output_shape": list(output.shape),
            "inference_time_ms": elapsed * 1000
        }

        print(f"Batch={batch_size}: 输入 {list(dummy_input.shape)} -> 输出 {list(output.shape)}, 耗时 {elapsed*1000:.2f}ms")

    # 参数量统计
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["parameters"] = {
        "total": num_params,
        "trainable": trainable_params
    }

    print(f"总参数：{num_params:,}, 可训练：{trainable_params:,}")

    return results


def test_text_module():
    """测试文本模块。"""
    print("\n" + "=" * 60)
    print("3. 文本模块测试")
    print("=" * 60)

    results = {}

    # 创建模型
    y_dim = 1000
    bit = 64
    model = DCMHTextModule(y_dim=y_dim, bit=bit)

    # 测试不同标签激活
    batch_size = 4
    test_cases = [
        ("单标签", 1),
        ("少标签", 5),
        ("多标签", 20),
    ]

    for case_name, num_active in test_cases:
        dummy_input = torch.zeros(batch_size, 1, y_dim, 1)
        for i in range(batch_size):
            active_indices = torch.randperm(y_dim)[:num_active]
            dummy_input[i, 0, active_indices, 0] = 1.0

        with torch.no_grad():
            output = model(dummy_input)

        results[case_name] = {
            "num_active_labels": num_active,
            "output_shape": list(output.shape),
            "output_range": [float(output.min()), float(output.max())]
        }

        print(f"{case_name} ({num_active} 标签): 输出 {list(output.shape)}, 范围 [{output.min():.4f}, {output.max():.4f}]")

    # 参数量
    num_params = sum(p.numel() for p in model.parameters())
    results["parameters"] = {"total": num_params}
    print(f"总参数：{num_params:,}")

    return results


def test_dual_stream_model():
    """测试双流模型。"""
    print("\n" + "=" * 60)
    print("4. 双流模型测试")
    print("=" * 60)

    results = {}

    # 创建模型
    bit = 64
    y_dim = 1000
    model = DCMHModel(bit=bit, y_dim=y_dim)

    # 准备输入
    batch_size = 4
    img_input = torch.randn(batch_size, 3, 224, 224)
    txt_input = torch.zeros(batch_size, 1, y_dim, 1)
    for i in range(batch_size):
        num_active = torch.randint(1, 10, (1,)).item()
        active = torch.randperm(y_dim)[:num_active]
        txt_input[i, 0, active, 0] = 1.0

    # 图像编码
    with torch.no_grad():
        img_hash = model.encode_image(img_input)
        txt_hash = model.encode_text(txt_input)
        img_out, txt_out = model(image=img_input, text=txt_input)
        binary_img = model.get_binary_hash('image', img_input)
        binary_txt = model.get_binary_hash('text', txt_input)

    results["encoding"] = {
        "image_hash_shape": list(img_hash.shape),
        "text_hash_shape": list(txt_hash.shape),
        "binary_image_unique": torch.unique(binary_img).tolist(),
        "binary_text_unique": torch.unique(binary_txt).tolist()
    }

    print(f"图像哈希：{list(img_hash.shape)}")
    print(f"文本哈希：{list(txt_hash.shape)}")
    print(f"二进制图像唯一值：{torch.unique(binary_img).tolist()}")
    print(f"二进制文本唯一值：{torch.unique(binary_txt).tolist()}")

    # 相似度计算
    similarity = model.compute_similarity(img_hash, txt_hash)
    results["similarity"] = {
        "matrix_shape": list(similarity.shape),
        "mean": float(similarity.mean()),
        "std": float(similarity.std()),
        "diag_mean": float(torch.diag(similarity).mean())
    }

    print(f"\n相似度矩阵：{list(similarity.shape)}, 均值：{similarity.mean():.4f}")

    # 汉明距离
    hamming_self = model.hamming_distance(binary_img, binary_img)
    results["hamming_self"] = float(hamming_self[0, 0])
    print(f"自汉明距离 [0,0]: {hamming_self[0, 0].item()}")

    # Top-K 检索
    distances, indices = model.cross_modal_retrieval(img_hash, txt_hash, top_k=3)
    results["retrieval"] = {
        "distances_shape": list(distances.shape),
        "indices_shape": list(indices.shape),
        "top3_distances": distances[0].tolist()
    }
    print(f"Top-3 检索距离：{distances[0].tolist()}")

    # 参数量
    num_params = sum(p.numel() for p in model.parameters())
    results["parameters"] = {"total": num_params}
    print(f"\n总参数：{num_params:,}")

    return results


def test_loss_functions():
    """测试损失函数。"""
    print("\n" + "=" * 60)
    print("5. 损失函数测试")
    print("=" * 60)

    results = {}

    batch_size = 8
    bit = 64
    img_hash = torch.randn(batch_size, bit)
    txt_hash = torch.randn(batch_size, bit)

    # 对比损失版本
    criterion_contrastive = DCMHCombinedLoss(
        margin=1.0,
        quantization_weight=0.1,
        use_info_nce=False
    )

    with torch.no_grad():
        total_loss, loss_dict = criterion_contrastive(img_hash, txt_hash)

    results["contrastive"] = {
        "total_loss": float(total_loss),
        "cross_modal": float(loss_dict["cross_modal"]),
        "quantization": float(loss_dict["quantization"])
    }

    print(f"对比损失版本:")
    print(f"  总损失：{total_loss.item():.4f}")
    print(f"  跨模态：{loss_dict['cross_modal'].item():.4f}")
    print(f"  量化：{loss_dict['quantization'].item():.4f}")

    # InfoNCE 版本
    criterion_nce = DCMHCombinedLoss(
        quantization_weight=0.1,
        use_info_nce=True,
        temperature=0.07
    )

    with torch.no_grad():
        total_loss_nce, loss_dict_nce = criterion_nce(img_hash, txt_hash)

    results["infonce"] = {
        "total_loss": float(total_loss_nce),
        "cross_modal": float(loss_dict_nce["cross_modal"]),
        "quantization": float(loss_dict_nce["quantization"])
    }

    print(f"\nInfoNCE 版本:")
    print(f"  总损失：{total_loss_nce.item():.4f}")
    print(f"  跨模态：{loss_dict_nce['cross_modal'].item():.4f}")
    print(f"  量化：{loss_dict_nce['quantization'].item():.4f}")

    # 单独量化损失
    q_loss_fn = DCMHQuantizationLoss()
    with torch.no_grad():
        q_loss = q_loss_fn(img_hash)
    results["quantization_only"] = float(q_loss)
    print(f"\n单独量化损失：{q_loss.item():.4f}")

    return results


def test_different_bit_configs():
    """测试不同哈希码长度配置。"""
    print("\n" + "=" * 60)
    print("6. 不同哈希码长度测试")
    print("=" * 60)

    results = {}

    for bit_config in [16, 32, 64, 128]:
        model = DCMHModel(bit=bit_config, y_dim=1000)

        batch_size = 2
        img_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            img_hash = model.encode_image(img_input)

        num_params = sum(p.numel() for p in model.parameters())

        results[f"bit_{bit_config}"] = {
            "bit": bit_config,
            "output_shape": list(img_hash.shape),
            "parameters": num_params
        }

        print(f"Bit={bit_config}: 输出 {list(img_hash.shape)}, 参数 {num_params:,}")

    return results


def generate_report(import_results, image_results, text_results,
                   dual_results, loss_results, bit_results):
    """生成 JSON 测试报告。"""

    report = {
        "test_date": datetime.now().isoformat(),
        "test_summary": {
            "total_tests": len(import_results),
            "passed": sum(1 for r in import_results if r["status"] == "PASS"),
            "failed": sum(1 for r in import_results if r["status"] == "FAIL")
        },
        "import_tests": import_results,
        "image_module_tests": image_results,
        "text_module_tests": text_results,
        "dual_stream_tests": dual_results,
        "loss_function_tests": loss_results,
        "bit_configuration_tests": bit_results,
        "config": DCMH_CONFIG
    }

    return report


def main():
    """主测试函数。"""
    print("=" * 60)
    print("DCMH 模型完整测试")
    print(f"测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 运行测试
    import_results = test_module_imports()
    image_results = test_image_module()
    text_results = test_text_module()
    dual_results = test_dual_stream_model()
    loss_results = test_loss_functions()
    bit_results = test_different_bit_configs()

    # 生成报告
    report = generate_report(
        import_results, image_results, text_results,
        dual_results, loss_results, bit_results
    )

    # 保存报告
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 保存 JSON 报告
    json_path = results_dir / "dcmh_test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON 报告已保存至：{json_path}")

    # 生成 Markdown 摘要
    md_summary = f"""# DCMH 模型测试摘要

**测试日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 测试概览

| 项目 | 结果 |
|------|------|
| 导入测试 | {report['test_summary']['passed']}/{report['test_summary']['total_tests']} 通过 |
| 图像模块参数 | {image_results['parameters']['total']:,} |
| 文本模块参数 | {text_results['parameters']['total']:,} |
| 双流模型参数 | {dual_results['parameters']['total']:,} |

## 模块测试结果

### 图像编码
- 输入：`[batch, 3, 224, 224]`
- 输出：`[batch, 64]`
- Batch=4 耗时：{image_results['batch_4']['inference_time_ms']:.2f}ms

### 文本编码
- 输入：`[batch, 1, 1000, 1]`
- 输出：`[batch, 64]`
- 支持标签数：1-20+

### 双流模型
- 图像哈希形状：{dual_results['encoding']['image_hash_shape']}
- 文本哈希形状：{dual_results['encoding']['text_hash_shape']}
- 二进制唯一值：{dual_results['encoding']['binary_image_unique']}
- 自汉明距离：{dual_results['hamming_self']}

### 损失函数
| 版本 | 总损失 | 跨模态 | 量化 |
|------|--------|--------|------|
| Contrastive | {loss_results['contrastive']['total_loss']:.4f} | {loss_results['contrastive']['cross_modal']:.4f} | {loss_results['contrastive']['quantization']:.4f} |
| InfoNCE | {loss_results['infonce']['total_loss']:.4f} | {loss_results['infonce']['cross_modal']:.4f} | {loss_results['infonce']['quantization']:.4f} |

## 不同哈希码长度

| Bit | 参数量 |
|-----|--------|
"""

    for bit_name, bit_data in bit_results.items():
        md_summary += f"| {bit_data['bit']} | {bit_data['parameters']:,} |\n"

    md_summary += f"""
## 配置信息

```json
{json.dumps(DCMH_CONFIG, indent=2)}
```

---

**详细结果**: 查看 `dcmh_test_results.json`
"""

    md_path = results_dir / "dcmh_test_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_summary)
    print(f"✓ Markdown 摘要已保存至：{md_path}")

    # 测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"导入测试：{report['test_summary']['passed']}/{report['test_summary']['total_tests']} 通过")
    print(f"图像模块参数：{image_results['parameters']['total']:,}")
    print(f"文本模块参数：{text_results['parameters']['total']:,}")
    print(f"双流模型参数：{dual_results['parameters']['total']:,}")
    print(f"对比损失：{loss_results['contrastive']['total_loss']:.4f}")
    print(f"InfoNCE 损失：{loss_results['infonce']['total_loss']:.4f}")
    print("\n✓ 所有测试完成!")

    return report


if __name__ == "__main__":
    main()
