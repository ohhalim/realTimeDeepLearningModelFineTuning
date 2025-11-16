#!/usr/bin/env python3
"""
샘플 MIDI 파일 생성 유틸리티

Brad Mehldau MIDI 데이터가 없을 때 테스트용 샘플 생성

사용법:
    python scripts/generate_sample_midi.py --output data/raw_midi --count 5
"""

import argparse
import os
import random
from pathlib import Path

import pretty_midi
import numpy as np


class JazzMIDIGenerator:
    """
    간단한 재즈 스타일 MIDI 생성기
    """

    def __init__(self, tempo=120):
        self.tempo = tempo

        # 재즈 코드 진행 (ii-V-I 등)
        self.chord_progressions = [
            # ii-V-I in C
            [(2, 5, 9), (7, 11, 2), (0, 4, 7)],  # Dm7, G7, Cmaj7
            # I-VI-ii-V
            [(0, 4, 7), (9, 0, 4), (2, 5, 9), (7, 11, 2)],
            # Blues progression
            [(0, 4, 7), (0, 4, 7), (0, 4, 7), (0, 4, 7),
             (5, 9, 0), (5, 9, 0), (0, 4, 7), (0, 4, 7),
             (7, 11, 2), (5, 9, 0), (0, 4, 7), (7, 11, 2)],
        ]

        # 재즈 스케일 (C major scale with blue notes)
        self.scale = [0, 2, 3, 4, 5, 7, 9, 10, 11]  # C, D, Eb, E, F, G, A, Bb, B

    def generate_chords(self, midi, duration=60):
        """코드 진행 생성"""
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        progression = random.choice(self.chord_progressions)
        bars = int(duration / (60 / self.tempo * 4))  # 4박자 기준

        current_time = 0
        chord_duration = 60 / self.tempo * 4  # 한 마디

        for bar in range(bars):
            chord = progression[bar % len(progression)]

            # 코드 보이싱 (루트 + 컬러 톤)
            root_octave = 3
            notes_pitches = [
                60 + chord[0] + (root_octave - 5) * 12,  # Root
                60 + chord[1] + (root_octave - 5) * 12,  # 3rd or extension
                60 + chord[2] + (root_octave - 5) * 12,  # 5th or extension
            ]

            # 약간의 변화를 위한 리듬 패턴
            if random.random() > 0.3:
                # 전체 코드
                for pitch in notes_pitches:
                    note = pretty_midi.Note(
                        velocity=random.randint(60, 80),
                        pitch=pitch,
                        start=current_time,
                        end=current_time + chord_duration * 0.95
                    )
                    piano.notes.append(note)
            else:
                # 쪼개진 코드 (컴핑 스타일)
                for i in range(2):
                    for pitch in notes_pitches:
                        note = pretty_midi.Note(
                            velocity=random.randint(50, 70),
                            pitch=pitch,
                            start=current_time + i * chord_duration / 2,
                            end=current_time + (i + 1) * chord_duration / 2 * 0.9
                        )
                        piano.notes.append(note)

            current_time += chord_duration

        midi.instruments.append(piano)

    def generate_melody(self, midi, duration=60):
        """멜로디 라인 생성"""
        piano = pretty_midi.Instrument(program=0)

        current_time = 0
        beat_duration = 60 / self.tempo

        while current_time < duration:
            # 음표 길이 (8분음표, 4분음표, 점4분음표 등)
            note_lengths = [
                beat_duration / 2,  # 8분음표
                beat_duration,      # 4분음표
                beat_duration * 1.5,  # 점4분음표
                beat_duration * 2,  # 2분음표
            ]
            note_length = random.choice(note_lengths)

            # 휴식 추가 (재즈의 공간감)
            if random.random() > 0.7:
                current_time += note_length
                continue

            # 스케일에서 음 선택
            scale_degree = random.choice(self.scale)
            octave = random.randint(5, 6)
            pitch = 60 + scale_degree + (octave - 5) * 12

            # 벨로시티 변화 (다이나믹스)
            velocity = random.randint(70, 100)

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=current_time,
                end=min(current_time + note_length * 0.9, duration)
            )
            piano.notes.append(note)

            current_time += note_length

        midi.instruments.append(piano)

    def generate_bass(self, midi, duration=60):
        """베이스 라인 생성"""
        bass = pretty_midi.Instrument(program=32)  # Acoustic Bass

        current_time = 0
        beat_duration = 60 / self.tempo

        # Walking bass 패턴
        while current_time < duration:
            # 4비트마다 베이스 음
            for _ in range(4):
                if current_time >= duration:
                    break

                # 낮은 음역의 스케일
                scale_degree = random.choice(self.scale)
                pitch = 36 + scale_degree  # 낮은 C 기준

                note = pretty_midi.Note(
                    velocity=random.randint(80, 95),
                    pitch=pitch,
                    start=current_time,
                    end=current_time + beat_duration * 0.9
                )
                bass.notes.append(note)

                current_time += beat_duration

        midi.instruments.append(bass)

    def generate(self, duration=60, complexity='medium'):
        """완전한 MIDI 파일 생성"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        if complexity in ['medium', 'high']:
            self.generate_chords(midi, duration)

        if complexity in ['high']:
            self.generate_melody(midi, duration)

        if complexity in ['medium', 'high']:
            self.generate_bass(midi, duration)

        return midi


def main():
    parser = argparse.ArgumentParser(
        description="샘플 재즈 MIDI 파일 생성"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_midi",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="생성할 파일 개수"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="각 파일 길이 (초)"
    )

    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("샘플 재즈 MIDI 생성")
    print("=" * 60)
    print(f"출력 디렉토리: {args.output}")
    print(f"파일 개수: {args.count}")
    print(f"파일 길이: {args.duration}초")
    print()

    complexities = ['medium', 'high']
    tempos = [100, 110, 120, 130, 140]

    for i in range(args.count):
        tempo = random.choice(tempos)
        complexity = random.choice(complexities)

        print(f"생성 중 {i+1}/{args.count}: "
              f"템포={tempo} BPM, 복잡도={complexity}...", end=" ")

        generator = JazzMIDIGenerator(tempo=tempo)
        midi = generator.generate(
            duration=args.duration,
            complexity=complexity
        )

        # 파일 저장
        output_path = os.path.join(
            args.output,
            f"sample_jazz_{i+1:02d}_tempo{tempo}.mid"
        )
        midi.write(output_path)

        print(f"✓ {output_path}")

    print()
    print("=" * 60)
    print("완료!")
    print(f"총 {args.count}개 파일 생성")
    print("=" * 60)


if __name__ == "__main__":
    main()
