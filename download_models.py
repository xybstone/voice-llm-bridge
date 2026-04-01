#!/usr/bin/env python3
"""
下载 VibeVoice 模型 (通过代理)
"""

import os
import sys

# 配置代理 (如果在国内)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    """下载模型"""
    print(f"\n📥 下载 {repo_id}...")
    print(f"   保存到：{local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"   ✅ 下载完成!")
        return True
    except Exception as e:
        print(f"   ❌ 下载失败：{e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VibeVoice 模型下载")
    print("=" * 60)
    
    # 模型列表
    models = [
        ("microsoft/VibeVoice-Realtime-0.5B", "~/models/VibeVoice-Realtime-0.5B"),
        # ("microsoft/VibeVoice-ASR", "~/models/VibeVoice-ASR"),  # 7B 太大，先不下载
    ]
    
    results = []
    for repo_id, local_dir in models:
        local_dir = os.path.expanduser(local_dir)
        success = download_model(repo_id, local_dir)
        results.append((repo_id, success))
    
    print("\n" + "=" * 60)
    print("下载总结:")
    for repo_id, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {repo_id}")
    print("=" * 60)
