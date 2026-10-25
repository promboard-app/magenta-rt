"""README 그림: 소리 파일 → 스펙트로그램(세로 = 주파수 0~12kHz 로그 크기, 가로 = 시간, 글자 없음) PNG.

python spectrogram.py <in.wav> <out.png> [가로 px 1000] [세로 px 260]
"""
import sys

import numpy as np
import soundfile as sf
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
H = int(sys.argv[4]) if len(sys.argv) > 4 else 260
x, sr = sf.read(src, always_2d=True)
m = x.mean(axis=1)
n_fft, hop = 2048, max(1, len(m) // W)
win = np.hanning(n_fft)
cols = []
for i in range(W):
    seg = m[i * hop: i * hop + n_fft]
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    cols.append(np.abs(np.fft.rfft(seg * win)))
S = np.array(cols).T  # [freq, time]
f = np.fft.rfftfreq(n_fft, 1 / sr)
S = S[f <= 12000]
db = 20 * np.log10(S + 1e-6)
db = np.clip((db - (db.max() - 80)) / 80, 0, 1)
# 주파수 축을 세로 H 칸으로(아래가 낮은 소리)
idx = np.linspace(0, db.shape[0] - 1, H).astype(int)
img = db[idx][::-1]
# 어두운 남색 → 보라 → 주황 → 연노랑
stops = np.array([[12, 14, 38], [74, 36, 120], [196, 72, 92], [246, 160, 60], [252, 240, 180]], np.float32)
pos = img * (len(stops) - 1)
lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
t = (pos - lo)[..., None]
rgb = stops[lo] * (1 - t) + stops[lo + 1] * t
Image.fromarray(rgb.astype(np.uint8)).save(dst)
print("ok", dst, (W, H))
