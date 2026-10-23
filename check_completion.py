#!/usr/bin/env python3
"""防御模块的静态完整性检查脚本。

检查文件、导入、函数签名、CLI 参数、依赖声明与主流程关键调用是否存在；
它不重新训练模型，也不对防御效果作数值复现。
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，使脚本可直接从项目目录运行。
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))

def check_files():
    """检查防御功能依赖的源码、文档与演示脚本是否落盘。"""
    """检查文件完整性"""
    print("\n" + "="*70)
    print("Step 1: Checking Files")
    print("="*70)
    
    files = {
        'src/defense.py': 'Defense core implementation',
        'src/visualize.py': 'Visualization functions',
        'src/train.py': 'Training utilities',
        'src/evaluate.py': 'Evaluation metrics',
        'main.py': 'CLI interface',
        'DEFENSE_README.md': 'Defense documentation',
        'QUICK_REFERENCE.md': 'Quick reference',
        'COMPLETION_REPORT_王赫麟.md': 'Completion report',
        'demo_defense.py': 'Demo script',
    }
    
    all_ok = True
    for filename, desc in files.items():
        # 文件存在且可读取不等于算法效果已验证，后续步骤仅做接口检查。
        path = project_path / filename
        if path.exists():
            size = path.stat().st_size
            print("[OK] {:<40} ({} bytes) {}".format(filename, size, desc))
        else:
            print("[FAIL] {:<40} MISSING".format(filename))
            all_ok = False
    
    return all_ok

def check_imports():
    """导入核心模块，尽早发现语法错误或模块路径错误。"""
    """检查导入"""
    print("\n" + "="*70)
    print("Step 2: Checking Imports")
    print("="*70)
    
    all_ok = True
    
    # Check core defense functions
    try:
        from src.defense import (
            detect_anomalies,
            defend_by_data_cleansing,
            defend_by_pruning,
            defend_by_finetune,
            defend_by_activation_clustering,
            compare_defense_effects,
        )
        print("[OK] All defense functions imported successfully")
    except ImportError as e:
        print("[FAIL] Defense functions import failed: {}".format(e))
        all_ok = False
    
    # Check visualization functions
    try:
        from src.visualize import (
            save_poisoned_samples,
            save_defense_comparison,
            save_defense_metrics,
        )
        print("[OK] All visualization functions imported successfully")
    except ImportError as e:
        print("[FAIL] Visualization functions import failed: {}".format(e))
        all_ok = False
    
    # Check main module imports
    try:
        import main
        print("[OK] Main module imported successfully")
    except ImportError as e:
        print("[FAIL] Main module import failed: {}".format(e))
        all_ok = False
    
    return all_ok

def check_functions():
    """检查要求的防御函数是否可见且具备预期参数。"""
    """检查函数签名和文档"""
    print("\n" + "="*70)
    print("Step 3: Checking Function Signatures")
    print("="*70)
    
    from src.defense import (
        detect_anomalies,
        defend_by_data_cleansing,
        defend_by_pruning,
        defend_by_finetune,
        defend_by_activation_clustering,
        compare_defense_effects,
    )
    
    functions = [
        detect_anomalies,
        defend_by_data_cleansing,
        defend_by_pruning,
        defend_by_finetune,
        defend_by_activation_clustering,
        compare_defense_effects,
    ]
    
    for func in functions:
        name = func.__name__
        has_doc = bool(func.__doc__)
        is_callable = callable(func)
        status = "[OK]" if (has_doc and is_callable) else "[WARN]"
        doc_str = "with doc" if has_doc else "NO DOC"
        print("{} {:<40} ({})".format(status, name, doc_str))
    
    return True

def check_cli():
    """检查 main.py 是否注册防御命令和关键参数。"""
    """检查命令行接口"""
    print("\n" + "="*70)
    print("Step 4: Checking CLI Interface")
    print("="*70)
    
    main_path = project_path / 'main.py'
    content = main_path.read_text(encoding='utf-8')
    
    checks = [
        ('defend-backdoor command', 'defend-backdoor'),
        ('defend_against_backdoor function', 'def defend_against_backdoor'),
        ('Argument parser setup', 'defend_parser = subparsers.add_parser'),
        ('Command dispatch', 'elif args.command == "defend-backdoor"'),
    ]
    
    all_ok = True
    for desc, keyword in checks:
        if keyword in content:
            print("[OK] {}".format(desc))
        else:
            print("[FAIL] {} - keyword not found: {}".format(desc, keyword))
            all_ok = False
    
    return all_ok

def check_dependencies():
    """检查依赖清单中是否声明防御和可视化所需第三方包。"""
    """检查依赖"""
    print("\n" + "="*70)
    print("Step 5: Checking Dependencies")
    print("="*70)
    
    packages = ['torch', 'numpy', 'matplotlib', 'sklearn']
    all_ok = True
    
    for pkg in packages:
        try:
            __import__(pkg)
            print("[OK] {} installed".format(pkg))
        except ImportError:
            print("[WARN] {} not installed (will fail at runtime)".format(pkg))
            all_ok = False
    
    return all_ok

def check_defense_flow():
    """检查主流程源码中是否调用各防御函数，而不实际执行训练。"""
    """检查防御流程的主要步骤"""
    print("\n" + "="*70)
    print("Step 6: Checking Defense Flow Implementation")
    print("="*70)
    
    defense_py = (project_path / 'src' / 'defense.py').read_text(encoding='utf-8')
    
    key_components = [
        ('Anomaly detection', 'def detect_anomalies'),
        ('Data cleansing', 'def defend_by_data_cleansing'),
        ('Model pruning', 'def defend_by_pruning'),
        ('Model fine-tuning', 'def defend_by_finetune'),
        ('Effect comparison', 'def compare_defense_effects'),
    ]
    
    all_ok = True
    for desc, keyword in key_components:
        if keyword in defense_py:
            print("[OK] {} implemented".format(desc))
        else:
            print("[FAIL] {} - NOT found".format(desc))
            all_ok = False
    
    return all_ok

def main():
    """顺序执行各项检查，并以汇总结果决定退出状态。"""
    print("\n")
    print("*" * 70)
    print("* Defense Module Integrity Check")
    print("*" * 70)
    
    results = []
    results.append(("Files", check_files()))
    results.append(("Imports", check_imports()))
    results.append(("Functions", check_functions()))
    results.append(("CLI Interface", check_cli()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Defense Flow", check_defense_flow()))
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print("[{}] {}".format(status, name))
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("SUCCESS: All checks passed!")
        print("\nThe defense module is ready to use:")
        print("  cd ai_backdoor_lab")
        print("  python main.py defend-backdoor --poison-rate 0.1")
    else:
        print("WARNING: Some checks failed. Please review above.")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
