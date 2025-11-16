#!/usr/bin/env python3
"""
Music Transformer 파인튜닝 스크립트

사용법:
    python scripts/train.py
"""

import os
import argparse
from datetime import datetime


# 설정
TFRECORD_DIR = "data/tfrecord"
PRETRAINED_MODEL = "models/pretrained/unconditional_model_16.ckpt"
OUTPUT_DIR = "models/finetuned/brad_mehldau"

# 하이퍼파라미터
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_TRAIN_STEPS = 10000
STEPS_PER_CHECKPOINT = 500
SEQUENCE_LENGTH = 2048


def check_prerequisites():
    """
    학습 전 필수 조건 확인
    """
    print("=" * 60)
    print("필수 조건 확인...")
    print("=" * 60)

    issues = []

    # 1. TFRecord 데이터 확인
    train_data = os.path.join(TFRECORD_DIR, "train", "train.tfrecord")
    if not os.path.exists(train_data):
        issues.append(f"학습 데이터 없음: {train_data}")
        issues.append("  → python scripts/preprocess.py 를 먼저 실행하세요")
    else:
        print(f"✓ 학습 데이터 존재: {train_data}")

    # 2. 사전학습 모델 확인
    if not os.path.exists(PRETRAINED_MODEL):
        print(f"⚠️  사전학습 모델 없음: {PRETRAINED_MODEL}")
        print("   → 처음부터 학습합니다 (시간이 더 걸릴 수 있음)")
    else:
        print(f"✓ 사전학습 모델 존재: {PRETRAINED_MODEL}")

    # 3. GPU 확인
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU 사용 가능: {len(gpus)}개")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("⚠️  GPU 없음 (CPU로 학습 - 매우 느림)")
    except Exception as e:
        print(f"⚠️  TensorFlow GPU 확인 실패: {e}")

    print("=" * 60 + "\n")

    if issues:
        print("❌ 오류:")
        for issue in issues:
            print(f"  {issue}")
        return False

    return True


def train_music_transformer():
    """
    Music Transformer 파인튜닝 실행

    Note: 이 함수는 실제 Magenta CLI 명령을 사용합니다.
    """
    print("=" * 60)
    print("Music Transformer 파인튜닝 시작")
    print("=" * 60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("하이퍼파라미터:")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  학습률: {LEARNING_RATE}")
    print(f"  학습 스텝: {NUM_TRAIN_STEPS}")
    print(f"  체크포인트 간격: {STEPS_PER_CHECKPOINT}")
    print(f"  시퀀스 길이: {SEQUENCE_LENGTH}")
    print()

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Magenta Music Transformer 학습 명령
    cmd = f"""
    music_transformer_train \\
      --config='unconditional' \\
      --data_dir={TFRECORD_DIR}/train \\
      --output_dir={OUTPUT_DIR} \\
      --restore_checkpoint={PRETRAINED_MODEL} \\
      --learning_rate={LEARNING_RATE} \\
      --batch_size={BATCH_SIZE} \\
      --num_train_steps={NUM_TRAIN_STEPS} \\
      --steps_per_checkpoint={STEPS_PER_CHECKPOINT} \\
      --sequence_length={SEQUENCE_LENGTH}
    """

    print("실행 명령:")
    print(cmd)
    print()

    print("=" * 60)
    print("⚠️  주의: 실제 학습은 Magenta CLI를 직접 실행해야 합니다")
    print("=" * 60)
    print()
    print("다음 명령을 터미널에서 실행하세요:")
    print()
    print(cmd.strip())
    print()
    print("또는 TensorBoard로 학습 모니터링:")
    print(f"  tensorboard --logdir={OUTPUT_DIR}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Brad Mehldau 스타일 Music Transformer 파인튜닝"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"배치 크기 (기본값: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"학습률 (기본값: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=NUM_TRAIN_STEPS,
        help=f"학습 스텝 (기본값: {NUM_TRAIN_STEPS})"
    )

    args = parser.parse_args()

    # 설정 업데이트
    global BATCH_SIZE, LEARNING_RATE, NUM_TRAIN_STEPS
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_TRAIN_STEPS = args.num_steps

    # 필수 조건 확인
    if not check_prerequisites():
        return

    # 학습 실행
    train_music_transformer()


if __name__ == "__main__":
    main()
