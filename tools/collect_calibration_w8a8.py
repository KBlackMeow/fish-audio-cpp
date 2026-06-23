#!/usr/bin/env python3
"""Collect a richer W8A8 calibration set with a persistent GPU server.

This script keeps `fish-server` alive in FP16 mode, sends a diverse mix of
plain-TTS and with-reference requests, and accumulates activation dumps under
`FISH_CALIBRATE_DIR`.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


DEFAULT_PROMPTS = [
    "今天天气不错，我们先确认一下语音的清晰度、停顿和句尾收束是否自然。",
    "请在早上八点半前提醒我参加产品评审会议，并把会议链接转发到项目群里。",
    "下面播报一则简讯：本季度平台整体可用率保持稳定，核心链路延迟较上月下降了百分之十二。",
    "她停了一下，轻声说道：别着急，我们还有时间，把每一步都做好就行。",
    "欢迎致电客户支持中心，如需查询订单进度，请先准备好手机号和订单编号。",
    "今天的直播到这里就告一段落，喜欢的话记得点赞、关注，并打开消息提醒。",
    "请把收货地址写成：上海市浦东新区张江路一百八十八号，三号楼十二层，前台代收。",
    "这段内容需要念得更像旁白，语速平稳一些，但关键字要稍微强调。",
    "版本号已经更新到 v2.3.17，预计在 2026 年 6 月 30 日晚间完成灰度发布。",
    "Hello everyone, welcome back to our channel. 今天我们继续测试中英混读时的连贯性和重音变化。",
    "如果你看到这条消息，请回复收到；如果暂时处理不了，也请先同步一下预计时间。",
    "她笑着问我，真的要现在出发吗？外面还在下雨，而且风也有一点大。",
    "本次演示包含数字、英文缩写和专有名词，例如 API、CUDA、JSON 以及 HTTP server。",
    "列车即将进站，请站在黄色安全线以内，先下后上，注意脚下间隙。",
    "接下来是一段更长的说明文本，用来覆盖连续句子、逗号停顿、括号补充以及末尾语气变化，确保模型在复杂文本场景下也能保持稳定自然的表达。",
    "用户反馈说，音色已经比较接近原版了，但语气和情绪起伏还是不够一致，这一点我们需要重点观察。",
    "请生成一段客服播报风格的语音，要求语气礼貌、信息清楚、节奏均匀，不要显得过于机械。",
    "这是一段偏口语化的内容：嗯，怎么说呢，其实整体还可以，就是有几个地方听起来没那么像本人。",
    "東京駅から品川駅まではおよそ十分です。次の予定に間に合うよう、少し早めに出発しましょう。",
    "请把这句话念得更坚定一些：我们会按时完成交付，并对最终质量负责。",
]


DEFAULT_REF_FILES = [
    ("example/vo_LLZAQ001_4_nahida_03.wav", "example/vo_LLZAQ001_4_nahida_03.lab"),
    ("example/000047.wav", "example/000047.lab"),
]


def load_ref_entry(root: Path, wav_rel: str, text_rel: str):
    wav_path = root / wav_rel
    text_path = root / text_rel
    if not wav_path.exists() or not text_path.exists():
        return None
    return {
        "ref_audio": base64.b64encode(wav_path.read_bytes()).decode("ascii"),
        "ref_audio_format": "wav",
        "ref_text": text_path.read_text(encoding="utf-8").strip(),
        "wav_path": str(wav_path),
    }


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            requests.post(
                f"{base_url}/v1/tts",
                json={"text": "ping", "max_new_tokens": 1},
                timeout=10,
            )
            return
        except Exception as exc:
            last_err = exc
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for server: {last_err}")


def summarize_calib_dir(calib_dir: Path) -> str:
    dump_files = sorted(calib_dir.glob("L*.bin"))
    total_bytes = sum(p.stat().st_size for p in dump_files)
    return f"{len(dump_files)} dump files, {total_bytes / (1024 * 1024):.1f} MB"


def main():
    parser = argparse.ArgumentParser(description="Collect richer W8A8 calibration dumps")
    parser.add_argument("--calib-dir", required=True, help="Output directory for activation dumps")
    parser.add_argument("--binary", default="./build/fish-server", help="Path to fish-server binary")
    parser.add_argument("--model-dir", default="models/s2-pro-fp16", help="FP16 model directory")
    parser.add_argument("--dtype", default="fp16", help="Model dtype for collection, usually fp16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--plain-count", type=int, default=12, help="How many plain TTS prompts to send")
    parser.add_argument("--ref-count", type=int, default=8, help="How many with-ref prompts to send")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep-existing", action="store_true", help="Append to existing dumps instead of cleaning the dir")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override")
    parser.add_argument("--merge-after", action="store_true", help="Merge small dump files after collection")
    parser.add_argument("--delete-raw-after-merge", action="store_true", help="Delete raw L*.bin files after merge")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    calib_dir = Path(args.calib_dir).resolve()
    calib_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for path in calib_dir.glob("L*.bin"):
            path.unlink()

    ref_entries = [
        entry for entry in
        (load_ref_entry(repo_root, wav_rel, txt_rel) for wav_rel, txt_rel in DEFAULT_REF_FILES)
        if entry is not None
    ]
    if not ref_entries:
        print("Warning: no reference audio found; with-ref calibration requests will be skipped", file=sys.stderr)
        args.ref_count = 0

    base_url = f"http://{args.host}:{args.port}"
    env = os.environ.copy()
    env["FISH_CALIBRATE_DIR"] = str(calib_dir)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        args.binary,
        "--server",
        "--host", args.host,
        "--port", str(args.port),
        "--model-dir", args.model_dir,
        "--dtype", args.dtype,
    ]
    print("Starting calibration server:")
    print("  " + " ".join(cmd))
    server_log = open(calib_dir / "fish_calibration_server.log", "w", encoding="utf-8")
    server = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    try:
        wait_for_server(base_url, args.timeout)
        print(f"Server ready at {base_url}")

        session = requests.Session()
        request_idx = 0

        for i in range(args.plain_count):
            text = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            payload = {
                "text": text,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed_base + request_idx,
            }
            r = session.post(f"{base_url}/v1/tts", json=payload, timeout=args.timeout)
            r.raise_for_status()
            print(f"[plain {i + 1}/{args.plain_count}] ok")
            request_idx += 1

        for i in range(args.ref_count):
            text = DEFAULT_PROMPTS[(i + args.plain_count) % len(DEFAULT_PROMPTS)]
            ref = ref_entries[i % len(ref_entries)]
            payload = {
                "text": text,
                "ref_audio": ref["ref_audio"],
                "ref_audio_format": ref["ref_audio_format"],
                "ref_text": ref["ref_text"],
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed_base + request_idx,
            }
            r = session.post(f"{base_url}/v1/tts/with-ref", json=payload, timeout=args.timeout)
            r.raise_for_status()
            print(f"[with-ref {i + 1}/{args.ref_count}] ok ({Path(ref['wav_path']).name})")
            request_idx += 1

        print("Calibration collection complete:")
        print(f"  {summarize_calib_dir(calib_dir)}")
        if args.merge_after:
            merge_cmd = [
                sys.executable,
                str(repo_root / "tools" / "merge_calibration_dumps.py"),
                str(calib_dir),
            ]
            if not args.delete_raw_after_merge:
                merge_cmd.append("--keep-originals")
            print("Merging calibration dumps:")
            print("  " + " ".join(merge_cmd))
            subprocess.run(merge_cmd, cwd=repo_root, check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        server_log.close()


if __name__ == "__main__":
    main()
