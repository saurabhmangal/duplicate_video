"""
video_processor.py
==================
Core engine for detecting duplicate and partial-match videos.

ALGORITHM OVERVIEW
------------------
1. FRAME SAMPLING
   - Each video is sampled at `sample_fps` frames/second (default: 1 fps)
   - This gives us a manageable set of representative frames

2. PERCEPTUAL HASHING (pHash)
   - Each sampled frame is converted to a 64-bit perceptual hash
   - pHash = Discrete Cosine Transform (DCT) of a 32x32 grayscale image
   - Two visually similar frames produce hashes with low Hamming distance
   - Hamming distance = number of differing bits (0=identical, 64=opposite)

3. VIDEO FINGERPRINT
   - A video's fingerprint = ordered list of pHashes (one per sampled frame)
   - Example: 5-minute video @ 1fps → array of ~300 hashes

4. EXACT DUPLICATE DETECTION
   - Two videos with similar lengths are compared hash-by-hash
   - avg_hamming = mean(hamming(A[i], B[i]) for all i)
   - avg_hamming ≤ threshold → DUPLICATE (same content, possibly different encoding/quality)

5. PARTIAL VIDEO DETECTION (Sliding Window)
   - Take shorter video A (length N hashes) and longer video B (length M hashes)
   - Slide a window of size N across B:
         for i in 0..M-N:
             window = B[i : i+N]
             dist   = mean(hamming(A[j], window[j]) for j in 0..N)
   - Find position i where dist is minimized
   - If min(dist) ≤ threshold → A is a SEGMENT of B starting at time i/sample_fps
"""

import cv2
import numpy as np
from PIL import Image
import imagehash
import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class VideoFingerprint:
    """Represents a video's extracted fingerprint."""
    id: str                          # Unique ID (filename used as ID)
    filename: str
    filepath: str
    duration: float                  # seconds
    width: int
    height: int
    frame_count: int
    hash_count: int                  # number of sampled frames
    hashes: List                     # list of imagehash.ImageHash objects
    thumbnail_b64: Optional[str]     # JPEG thumbnail as base64 string


@dataclass
class ComparisonResult:
    """Result of comparing two videos."""
    type: str                        # "duplicate" | "partial"
    video1_id: str
    video2_id: str
    similarity_pct: float            # 0–100%
    avg_hamming: float
    # Only for partial matches:
    segment_start_sec: Optional[float] = None
    segment_end_sec: Optional[float] = None
    shorter_id: Optional[str] = None
    longer_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Video Processor
# ─────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Processes videos and detects duplicates / partial matches.

    Parameters
    ----------
    sample_fps : float
        How many frames per second to sample (default 1.0).
        Higher = more accurate but slower.
    hash_size : int
        Size of the DCT matrix (hash_size x hash_size = total bits).
        Default 8 → 64-bit hash.
    dup_threshold : int
        Maximum avg Hamming distance to call two videos "duplicates".
        Out of 64 bits, ≤8 means ≥87.5% similarity.
    partial_threshold : int
        Same but for sliding-window partial match.
    min_partial_hashes : int
        Minimum number of hashes the shorter video must have
        before we attempt partial matching (avoids false positives
        from very short clips).
    length_ratio_tolerance : float
        How close in length two videos must be to be checked for
        exact duplicate (0.15 = within 15%).
    """

    def __init__(
        self,
        sample_fps: float = 1.0,
        hash_size: int = 8,
        dup_threshold: int = 8,
        partial_threshold: int = 8,
        min_partial_hashes: int = 10,
        length_ratio_tolerance: float = 0.15,
    ):
        self.sample_fps = sample_fps
        self.hash_size = hash_size
        self.dup_threshold = dup_threshold
        self.partial_threshold = partial_threshold
        self.min_partial_hashes = min_partial_hashes
        self.length_ratio_tolerance = length_ratio_tolerance
        self.max_bits = hash_size * hash_size  # 64 for default

    # ----------------------------------------------------------
    # STEP 1 — Extract fingerprint from a single video
    # ----------------------------------------------------------
    def extract_fingerprint(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> Optional[VideoFingerprint]:
        """
        Open a video file, sample frames at `sample_fps`, compute pHash
        for each frame, and return a VideoFingerprint.

        pHash Algorithm per frame:
          1. Resize frame to (hash_size*4) x (hash_size*4) grayscale
          2. Apply 2D DCT (Discrete Cosine Transform)
          3. Keep top-left hash_size x hash_size block (low frequencies)
          4. Compute mean value of that block
          5. 64-bit hash: bit[i] = 1 if pixel[i] > mean else 0
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # How many native video frames to skip between each sample
        sample_interval = max(1, int(fps / self.sample_fps))

        hashes = []
        thumbnail_b64 = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # Convert BGR (OpenCV) → RGB (PIL)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # ── Perceptual Hash ──────────────────────────────────
                # imagehash.phash internally does:
                #   resize → grayscale → DCT → threshold → 64-bit int
                ph = imagehash.phash(pil_img, hash_size=self.hash_size)
                hashes.append(ph)

                # Capture first frame as thumbnail
                if thumbnail_b64 is None:
                    thumb = pil_img.resize((320, 180), Image.LANCZOS)
                    buf = io.BytesIO()
                    thumb.save(buf, format="JPEG", quality=75)
                    thumbnail_b64 = base64.b64encode(buf.getvalue()).decode()

                if progress_cb and total_frames > 0:
                    progress_cb(frame_idx / total_frames)

            frame_idx += 1

        cap.release()

        if not hashes:
            return None

        return VideoFingerprint(
            id=os.path.basename(video_path),
            filename=os.path.basename(video_path),
            filepath=video_path,
            duration=duration,
            width=width,
            height=height,
            frame_count=total_frames,
            hash_count=len(hashes),
            hashes=hashes,
            thumbnail_b64=thumbnail_b64,
        )

    # ----------------------------------------------------------
    # STEP 2 — Compare two fingerprints
    # ----------------------------------------------------------
    def _avg_hamming(self, hashes_a: List, hashes_b: List) -> float:
        """Compute average Hamming distance between two equal-length hash lists."""
        if not hashes_a or not hashes_b:
            return float("inf")
        n = min(len(hashes_a), len(hashes_b))
        total = sum(hashes_a[i] - hashes_b[i] for i in range(n))
        return total / n

    def _sliding_window_search(
        self, short_h: List, long_h: List
    ) -> Tuple[float, int]:
        """
        Slide a window of len(short_h) over long_h.
        Returns (min_avg_hamming, best_start_index).

        WHY SLIDING WINDOW?
        If video A (3 min) is a clip from video B (30 min), the matching
        segment could start anywhere in B. We test every possible starting
        position and keep the one with lowest avg Hamming distance.
        """
        n = len(short_h)
        m = len(long_h)

        if n > m:
            return float("inf"), 0

        min_dist = float("inf")
        best_pos = 0

        for i in range(m - n + 1):
            window = long_h[i : i + n]
            dist = self._avg_hamming(short_h, window)
            if dist < min_dist:
                min_dist = dist
                best_pos = i
                # Early exit: perfect match found
                if min_dist == 0:
                    break

        return min_dist, best_pos

    def compare(
        self,
        fp1: VideoFingerprint,
        fp2: VideoFingerprint,
    ) -> Optional[ComparisonResult]:
        """
        Compare two VideoFingerprints.

        Decision tree:
          1. If lengths are similar (within tolerance):
               → Check for EXACT DUPLICATE via direct hash comparison
          2. If lengths differ significantly:
               → Check for PARTIAL MATCH via sliding window
          3. Even for similar-length videos that aren't exact duplicates:
               → Also check partial (one might be a trimmed version)
        """
        h1, h2 = fp1.hashes, fp2.hashes
        n1, n2 = len(h1), len(h2)

        if n1 == 0 or n2 == 0:
            return None

        max_len = max(n1, n2)
        length_ratio_diff = abs(n1 - n2) / max_len

        # ── A. EXACT DUPLICATE CHECK ────────────────────────────
        # Only makes sense when videos have roughly the same length
        if length_ratio_diff <= self.length_ratio_tolerance:
            avg_dist = self._avg_hamming(h1, h2)
            if avg_dist <= self.dup_threshold:
                similarity = (1 - avg_dist / self.max_bits) * 100
                return ComparisonResult(
                    type="duplicate",
                    video1_id=fp1.id,
                    video2_id=fp2.id,
                    similarity_pct=round(similarity, 1),
                    avg_hamming=round(avg_dist, 2),
                )

        # ── B. PARTIAL MATCH CHECK (sliding window) ─────────────
        # Determine shorter and longer video
        if n1 <= n2:
            shorter_fp, short_h, longer_fp, long_h = fp1, h1, fp2, h2
        else:
            shorter_fp, short_h, longer_fp, long_h = fp2, h2, fp1, h1

        # Skip if the shorter video is too brief (unreliable matching)
        if len(short_h) < self.min_partial_hashes:
            return None

        min_dist, best_pos = self._sliding_window_search(short_h, long_h)

        if min_dist <= self.partial_threshold:
            similarity = (1 - min_dist / self.max_bits) * 100
            seg_start = best_pos / self.sample_fps
            seg_end = (best_pos + len(short_h)) / self.sample_fps
            return ComparisonResult(
                type="partial",
                video1_id=fp1.id,
                video2_id=fp2.id,
                similarity_pct=round(similarity, 1),
                avg_hamming=round(min_dist, 2),
                segment_start_sec=round(seg_start, 1),
                segment_end_sec=round(seg_end, 1),
                shorter_id=shorter_fp.id,
                longer_id=longer_fp.id,
            )

        return None

    # ----------------------------------------------------------
    # STEP 3 — Analyze all videos in a list
    # ----------------------------------------------------------
    def analyze(
        self,
        video_paths: List[str],
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Dict:
        """
        Main entry point.
        1. Extract fingerprints from all videos.
        2. Compare every pair (O(n²) pairs).
        3. Return structured results dict.
        """
        fingerprints: List[VideoFingerprint] = []
        errors = []
        total = len(video_paths)

        # ── Phase 1: Fingerprint extraction ─────────────────────
        for idx, path in enumerate(video_paths):
            filename = os.path.basename(path)
            if progress_cb:
                progress_cb(idx, f"Extracting fingerprint: {filename}")
            try:
                fp = self.extract_fingerprint(path)
                if fp:
                    fingerprints.append(fp)
                else:
                    errors.append({"file": filename, "error": "Could not open video"})
            except Exception as e:
                errors.append({"file": filename, "error": str(e)})

        if progress_cb:
            progress_cb(total, "Fingerprints extracted. Comparing pairs…")

        # ── Phase 2: Pairwise comparison ─────────────────────────
        duplicates = []
        partials = []
        n = len(fingerprints)
        total_pairs = n * (n - 1) // 2
        pair_idx = 0

        for i in range(n):
            for j in range(i + 1, n):
                pair_idx += 1
                if progress_cb:
                    progress_cb(
                        total + pair_idx,
                        f"Comparing {fingerprints[i].filename} ↔ {fingerprints[j].filename}",
                    )
                result = self.compare(fingerprints[i], fingerprints[j])
                if result:
                    if result.type == "duplicate":
                        duplicates.append(
                            {
                                "video1": result.video1_id,
                                "video2": result.video2_id,
                                "similarity_pct": result.similarity_pct,
                                "avg_hamming": result.avg_hamming,
                            }
                        )
                    elif result.type == "partial":
                        partials.append(
                            {
                                "shorter_video": result.shorter_id,
                                "longer_video": result.longer_id,
                                "similarity_pct": result.similarity_pct,
                                "avg_hamming": result.avg_hamming,
                                "segment_start_sec": result.segment_start_sec,
                                "segment_end_sec": result.segment_end_sec,
                            }
                        )

        # ── Phase 3: Build response ──────────────────────────────
        return {
            "videos": [
                {
                    "id": fp.id,
                    "filename": fp.filename,
                    "duration_sec": round(fp.duration, 1),
                    "duration_fmt": _fmt_duration(fp.duration),
                    "resolution": f"{fp.width}×{fp.height}",
                    "hash_count": fp.hash_count,
                    "thumbnail": fp.thumbnail_b64,
                }
                for fp in fingerprints
            ],
            "duplicates": duplicates,
            "partials": partials,
            "errors": errors,
            "stats": {
                "total_videos": len(fingerprints),
                "total_pairs_checked": total_pairs,
                "duplicates_found": len(duplicates),
                "partials_found": len(partials),
            },
        }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """Format seconds → HH:MM:SS or MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"
