#!/usr/bin/env python3
"""
파인튜닝된 모델로 음악 생성 스크립트

사용법:
    python scripts/generate.py --checkpoint models/finetuned/brad_mehldau/model.ckpt-10000
"""

import os
import argparse
from datetime import datetime
import glob


# 설정
OUTPUT_DIR = "output"
DEFAULT_NUM_OUTPUTS = 5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_NUM_STEPS = 2048


def find_latest_checkpoint(checkpoint_dir):
    """
    가장 최근 체크포인트 찾기
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "model.ckpt-*.index"))
    if not checkpoints:
        return None

    # 스텝 번호로 정렬
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.split('-')[-1].replace('.index', ''))
    )

    latest = checkpoints[-1].replace('.index', '')
    return latest


def generate_music(checkpoint_path, num_outputs, temperature, num_steps,
                   primer_midi=None):
    """
    음악 생성

    Args:
        checkpoint_path: 모델 체크포인트 경로
        num_outputs: 생성할 MIDI 파일 수
        temperature: 샘플링 온도 (0.5~1.5)
        num_steps: 생성할 스텝 수
        primer_midi: 프라이머 MIDI 파일 (선택)
    """
    print("=" * 60)
    print("Brad Mehldau 스타일 음악 생성")
    print("=" * 60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("설정:")
    print(f"  체크포인트: {checkpoint_path}")
    print(f"  생성 개수: {num_outputs}")
    print(f"  Temperature: {temperature}")
    print(f"  스텝 수: {num_steps}")
    if primer_midi:
        print(f"  프라이머: {primer_midi}")
    print()

    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_subdir = os.path.join(OUTPUT_DIR, f"generation_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)

    # Magenta Music Transformer 생성 명령
    primer_flag = f"--primer_midi={primer_midi}" if primer_midi else ""

    cmd = f"""
    music_transformer_generate \\
      --config='unconditional' \\
      --checkpoint_file={checkpoint_path} \\
      --output_dir={output_subdir} \\
      --num_outputs={num_outputs} \\
      --temperature={temperature} \\
      --num_steps={num_steps} \\
      {primer_flag}
    """

    print("실행 명령:")
    print(cmd.strip())
    print()

    print("=" * 60)
    print("⚠️  주의: 실제 생성은 Magenta CLI를 직접 실행해야 합니다")
    print("=" * 60)
    print()
    print("다음 명령을 터미널에서 실행하세요:")
    print()
    print(cmd.strip())
    print()
    print(f"생성된 MIDI 파일은 {output_subdir}/ 에 저장됩니다.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="파인튜닝된 모델로 Brad Mehldau 스타일 음악 생성"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="모델 체크포인트 경로 (미지정 시 최신 체크포인트 자동 선택)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/finetuned/brad_mehldau",
        help="체크포인트 디렉토리 (기본값: models/finetuned/brad_mehldau)"
    )
    parser.add_argument(
        "--num_outputs",
        type=int,
        default=DEFAULT_NUM_OUTPUTS,
        help=f"생성할 MIDI 파일 수 (기본값: {DEFAULT_NUM_OUTPUTS})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"샘플링 온도 (기본값: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"생성할 스텝 수 (기본값: {DEFAULT_NUM_STEPS})"
    )
    parser.add_argument(
        "--primer",
        type=str,
        help="프라이머 MIDI 파일 경로 (선택)"
    )

    args = parser.parse_args()

    # 체크포인트 결정
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print(f"체크포인트 자동 선택 중... ({args.checkpoint_dir})")
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)

        if not checkpoint_path:
            print(f"❌ 체크포인트를 찾을 수 없습니다: {args.checkpoint_dir}")
            print("   먼저 모델을 학습하세요: python scripts/train.py")
            return

        print(f"✓ 선택된 체크포인트: {checkpoint_path}\n")

    # 음악 생성
    generate_music(
        checkpoint_path=checkpoint_path,
        num_outputs=args.num_outputs,
        temperature=args.temperature,
        num_steps=args.num_steps,
        primer_midi=args.primer
    )


if __name__ == "__main__":
    main()
