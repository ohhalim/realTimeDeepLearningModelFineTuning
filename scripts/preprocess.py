#!/usr/bin/env python3
"""
Brad Mehldau MIDI 파일 전처리 스크립트

사용법:
    python scripts/preprocess.py
"""

import os
import glob
import random
from typing import List, Tuple
import pretty_midi
from magenta.music import midi_io, sequences_lib
from magenta.pipelines import note_sequence_pipelines
import tensorflow as tf


# 설정
RAW_MIDI_DIR = "data/raw_midi"
PROCESSED_DIR = "data/processed"
TFRECORD_DIR = "data/tfrecord"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MIN_DURATION = 60  # 최소 60초
STEPS_PER_QUARTER = 4  # 16분음표 단위


def validate_midi_file(filepath: str) -> Tuple[bool, str]:
    """
    MIDI 파일의 유효성 검사

    Returns:
        (is_valid, message)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(filepath)
        duration = midi_data.get_end_time()

        if duration < MIN_DURATION:
            return False, f"너무 짧음 ({duration:.1f}초 < {MIN_DURATION}초)"

        # 피아노 트랙 확인
        has_piano = any(
            inst.program in range(0, 8) or 'piano' in inst.name.lower()
            for inst in midi_data.instruments
        )

        if not has_piano:
            return False, "피아노 트랙 없음"

        return True, f"유효 ({duration:.1f}초)"

    except Exception as e:
        return False, f"오류: {str(e)}"


def validate_all_midi_files(midi_dir: str) -> List[str]:
    """
    모든 MIDI 파일 검증

    Returns:
        유효한 파일 경로 리스트
    """
    print("=" * 60)
    print("MIDI 파일 검증 중...")
    print("=" * 60)

    midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + \
                 glob.glob(os.path.join(midi_dir, "*.midi"))

    valid_files = []

    for filepath in sorted(midi_files):
        filename = os.path.basename(filepath)
        is_valid, message = validate_midi_file(filepath)

        status = "✓" if is_valid else "✗"
        print(f"{status} {filename}: {message}")

        if is_valid:
            valid_files.append(filepath)

    print("\n" + "=" * 60)
    print(f"총 {len(midi_files)}개 중 {len(valid_files)}개 유효")
    print("=" * 60 + "\n")

    return valid_files


def convert_to_notesequence(midi_file: str) -> sequences_lib.NoteSequence:
    """
    MIDI 파일을 Magenta NoteSequence로 변환
    """
    # MIDI를 NoteSequence로 변환
    ns = midi_io.midi_file_to_note_sequence(midi_file)

    # 양자화 (16분음표 기준)
    qns = sequences_lib.quantize_note_sequence(ns, STEPS_PER_QUARTER)

    return qns


def split_dataset(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    데이터셋을 train/val/test로 분할
    """
    random.seed(42)
    random.shuffle(files)

    n_files = len(files)
    n_train = int(n_files * TRAIN_RATIO)
    n_val = int(n_files * VAL_RATIO)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"데이터 분할:")
    print(f"  학습: {len(train_files)}개")
    print(f"  검증: {len(val_files)}개")
    print(f"  테스트: {len(test_files)}개\n")

    return train_files, val_files, test_files


def save_tfrecord(note_sequences: List[sequences_lib.NoteSequence],
                  output_file: str):
    """
    NoteSequence를 TFRecord 파일로 저장
    """
    with tf.io.TFRecordWriter(output_file) as writer:
        for ns in note_sequences:
            writer.write(ns.SerializeToString())


def process_and_save(files: List[str], output_dir: str, split_name: str):
    """
    MIDI 파일들을 처리하고 TFRecord로 저장
    """
    print(f"\n{split_name} 데이터 처리 중...")

    note_sequences = []

    for filepath in files:
        try:
            ns = convert_to_notesequence(filepath)
            note_sequences.append(ns)
            print(f"  ✓ {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  ✗ {os.path.basename(filepath)}: {e}")

    # TFRecord로 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split_name}.tfrecord")
    save_tfrecord(note_sequences, output_file)

    print(f"  → 저장: {output_file} ({len(note_sequences)}개 시퀀스)")


def main():
    """
    메인 전처리 파이프라인
    """
    print("\n" + "=" * 60)
    print("Brad Mehldau MIDI 전처리 시작")
    print("=" * 60 + "\n")

    # 1. MIDI 파일 검증
    valid_files = validate_all_midi_files(RAW_MIDI_DIR)

    if len(valid_files) == 0:
        print("⚠️  유효한 MIDI 파일이 없습니다!")
        print(f"   {RAW_MIDI_DIR} 폴더에 MIDI 파일을 추가해주세요.")
        return

    # 2. 데이터 분할
    train_files, val_files, test_files = split_dataset(valid_files)

    # 3. 처리 및 저장
    process_and_save(train_files, os.path.join(TFRECORD_DIR, "train"), "train")
    process_and_save(val_files, os.path.join(TFRECORD_DIR, "val"), "val")
    process_and_save(test_files, os.path.join(TFRECORD_DIR, "test"), "test")

    print("\n" + "=" * 60)
    print("✓ 전처리 완료!")
    print("=" * 60)
    print("\n다음 단계: python scripts/train.py\n")


if __name__ == "__main__":
    main()
