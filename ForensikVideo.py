# ForensikVideo_DFRWS_Advanced.py
"""
Sistem Forensik Video Modular dengan Analisis Multi-Level Berbasis Kerangka DFRWS
==================================================================================
Versi canggih yang mendeteksi manipulasi temporal dan mengintegrasikan analisis
metadata encoder, serta kerangka kerja untuk deteksi PRNU dan GAN/Deepfake.
Menghasilkan laporan investigasi terpadu yang sangat detail dan visual.

Author : OpenAI-GPT & [Nama Anda] (Tahun)
License: MIT
"""
from __future__ import annotations
import argparse
import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import cv2
import imagehash
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, Table, TableStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################################################
# Utilitas Visualisasi DFRWS
###############################################################################
class Icons:
    IDENTIFICATION = "🔍"; PRESERVATION = "🛡️"; COLLECTION = "📥"; EXAMINATION = "🔬"
    ANALYSIS = "📈"; REPORTING = "📄"; SUCCESS = "✅"; ERROR = "❌"; INFO = "ℹ️"

def print_stage_banner(stage_name: str, icon: str, description: str):
    width = 80
    print("\n" + "=" * width); print(f"=== {icon}  TAHAP DFRWS: {stage_name.upper()} ".ljust(width - 3) + "===")
    print("=" * width); print(f"{Icons.INFO}  {description}"); print("-" * width)

###############################################################################
# Dataclass yang Ditingkatkan untuk Pelaporan Terpadu
###############################################################################
@dataclass
class FrameInfo:
    index: int; timestamp: float; img_path: str
    hash: str | None = None; type: str = "unknown"; evidence: dict | None = None

@dataclass
class VideoResult:
    video: str; preservation_hash_sha256: str; metadata: dict
    frames: list[FrameInfo]; summary: dict
    temporal_plot: str | None = None; localizations: list[dict] | None = None
    # DITAMBAHKAN: Untuk menyimpan hasil analisis lanjutan
    advanced_analysis: dict | None = None

@dataclass
class ComparisonResult:
    """DITAMBAHKAN: Struktur data untuk membungkus hasil analisis perbandingan."""
    baseline_result: VideoResult
    suspect_result: VideoResult

###############################################################################
# Modul Analisis Lanjutan (Praktis dan Konseptual)
###############################################################################

def analyze_encoder_metadata(metadata: dict) -> dict:
    """
    MODUL BARU (PRAKTIS): Mengekstrak metadata terkait encoder dan pembuatan.
    Ini dapat mengindikasikan proses re-encoding, sebuah jejak manipulasi.
    """
    findings = {}
    # Cari di tag format utama
    if 'format' in metadata and 'tags' in metadata['format']:
        for key, value in metadata['format']['tags'].items():
            if 'encoder' in key.lower() or 'software' in key.lower():
                findings['encoding_software'] = value
            if 'creation_time' in key.lower():
                findings['creation_time'] = value
    
    # Cari di setiap stream
    if 'streams' in metadata:
        for stream in metadata['streams']:
            if 'tags' in stream:
                for key, value in stream['tags'].items():
                    if 'encoder' in key.lower() and 'encoding_software' not in findings:
                        findings['encoding_software'] = value
                    if 'creation_time' in key.lower() and 'creation_time' not in findings:
                        findings['creation_time'] = value
    
    if not findings:
        return {"status": "Tidak ada metadata encoder/pembuatan yang jelas ditemukan."}
    return findings

def analyze_prnu_placeholder(frames: list[FrameInfo]) -> dict:
    """
    MODUL BARU (KONSEPTUAL): Placeholder untuk analisis PRNU (Sensor Noise).
    Implementasi nyata memerlukan pola referensi kamera (kamera yang sama) dan pustaka khusus.
    """
    # Di dunia nyata, di sini kita akan:
    # 1. Menghitung noise residual dari setiap bingkai.
    # 2. Menghitung korelasi silang (cross-correlation) antara noise residual dan pola PRNU referensi.
    # 3. Jika korelasi melebihi ambang batas, bingkai dianggap berasal dari kamera sumber.
    return {
        "status": "Modul Belum Diimplementasikan.",
        "description": "Analisis PRNU memerlukan pola noise referensi dari kamera sumber untuk memverifikasi keaslian.",
        "required_input": "File pola PRNU referensi (misal, .npy atau .mat)."
    }

def analyze_gan_deepfake_placeholder(frames: list[FrameInfo]) -> dict:
    """
    MODUL BARU (KONSEPTUAL): Placeholder untuk deteksi GAN/Deepfake.
    Implementasi nyata memerlukan model deep learning yang sudah dilatih.
    """
    # Di dunia nyata, di sini kita akan:
    # 1. Melakukan loop pada setiap bingkai atau bingkai kunci (key frames).
    # 2. Melewatkan setiap bingkai melalui model klasifikasi (misal, EfficientNet, ResNet) yang dilatih untuk membedakan gambar asli vs. buatan.
    # 3. Mengumpulkan skor kepercayaan (confidence scores) dan melaporkan bingkai yang mencurigakan.
    return {
        "status": "Modul Belum Diimplementasikan.",
        "description": "Deteksi Deepfake memerlukan model Deep Learning yang telah dilatih pada dataset gambar asli dan sintetis.",
        "required_input": "File model yang telah dilatih (misal, .pth, .h5, atau .onnx)."
    }

###############################################################################
# Fungsi Helper (Inti tidak berubah)
###############################################################################
def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""): sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e: print(f"{Icons.ERROR} Gagal membaca file: {e}", file=sys.stderr); return "error"

def ffprobe_metadata(path: str | Path) -> dict:
    cmd = ["ffprobe","-v","quiet","-print_format","json","-show_format","-show_streams",str(path),]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8'); return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError): print(f"{Icons.ERROR} ffprobe/ffmpeg error.", file=sys.stderr); sys.exit(1)
    except json.JSONDecodeError: print(f"{Icons.ERROR} Gagal parsing JSON ffprobe.", file=sys.stderr); return {"error": "parse_failed"}

def extract_frames_ffmpeg(path: str | Path, out_dir: Path, fps: int = 60) -> int:
    out_dir.mkdir(parents=True, exist_ok=True); pattern = str(out_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg","-i",str(path),"-vf",f"fps={fps}","-qscale:v","2",pattern,"-hide_banner","-loglevel","error",]
    try: subprocess.run(cmd, check=True); return len(list(out_dir.glob('*.jpg')))
    except (subprocess.CalledProcessError, FileNotFoundError): print(f"{Icons.ERROR} ffmpeg error.", file=sys.stderr); sys.exit(1)

def compute_hash(img_path: Path) -> str:
    img = Image.open(img_path).convert("L").resize((8, 8), Image.Resampling.LANCZOS); return str(imagehash.average_hash(img))

def hash_distance(h1: str, h2: str) -> int:
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)

def optical_flow_magnitude(prev_gray: np.ndarray, next_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1]); return float(np.mean(mag))

###############################################################################
# Algoritma Analisis (Inti tidak berubah)
###############################################################################
def analyze_pairwise(base_frames: list[FrameInfo], sus_frames: list[FrameInfo], hash_thresh: int = 6):
    i = j = 0
    while i < len(base_frames) and j < len(sus_frames):
        if hash_distance(base_frames[i].hash, sus_frames[j].hash) <= hash_thresh: sus_frames[j].type = "original"; i += 1; j += 1
        else:
            if j + 1 < len(sus_frames) and hash_distance(base_frames[i].hash, sus_frames[j + 1].hash) <= hash_thresh: sus_frames[j].type = "anomaly_insert"; sus_frames[j].evidence = {"description": f"Tidak ditemukan di referensi."}; j += 1
            elif i + 1 < len(base_frames) and hash_distance(base_frames[i + 1].hash, sus_frames[j].hash) <= hash_thresh: print(f"  {Icons.INFO} Deteksi penghapusan: bingkai referensi {i} hilang."); i += 1
            else: sus_frames[j].type = "anomaly_discontinuity"; sus_frames[j].evidence = {"description": f"Tidak cocok dengan ref. frame {i}."}; i += 1; j += 1
    for k in range(j, len(sus_frames)): sus_frames[k].type = "anomaly_insert"; sus_frames[k].evidence = {"description": "Bingkai tambahan di akhir."}
    if i < len(base_frames): print(f"  {Icons.INFO} {len(base_frames) - i} bingkai akhir referensi hilang.")

def analyze_single(frames: list[FrameInfo], hash_thresh_dup: int = 2, flow_z_thresh: float = 4.0):
    hashes = [f.hash for f in frames]; n = len(frames)
    if n < 2:
        if n == 1: frames[0].type = "original"; return
    for idx in range(1, n):
        if hash_distance(hashes[idx], hashes[idx - 1]) <= hash_thresh_dup: frames[idx].type = "anomaly_duplicate"; frames[idx].evidence = {"dup_of_index": idx - 1, "hash_distance": hash_distance(hashes[idx], hashes[idx - 1])}
    mags = []; print(f"  {Icons.INFO} Menganalisis aliran optik...");
    for idx in tqdm(range(1, n), desc="    Menghitung Aliran Optik", leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        if frames[idx].type != 'unknown': continue
        try:
            img_prev = cv2.imread(frames[idx - 1].img_path, cv2.IMREAD_GRAYSCALE); img_next = cv2.imread(frames[idx].img_path, cv2.IMREAD_GRAYSCALE)
            if img_prev is None or img_next is None: continue
            mags.append((idx, optical_flow_magnitude(img_prev, img_next)))
        except Exception as e: print(f"  {Icons.ERROR} Gagal aliran optik bingkai {idx}: {e}")
    if mags:
        flow_values = np.array([m[1] for m in mags]); median = np.median(flow_values); mad = np.median(np.abs(flow_values - median)) + 1e-9
        for idx, mag in mags:
            z_score = 0.6745 * (mag - median) / mad
            if abs(z_score) >= flow_z_thresh and frames[idx].type == 'unknown': frames[idx].type = "anomaly_discontinuity"; frames[idx].evidence = {"optical_flow_magnitude": float(mag), "z_score": float(z_score)}
    for f in frames:
        if f.type == 'unknown': f.type = "original"

###############################################################################
# Pelaporan Terpadu dan Visualisasi yang Diperkaya
###############################################################################
def plot_temporal_anomalies(frames: list[FrameInfo], out_path: Path):
    if not frames: return
    idxs = [f.index for f in frames]; labels = [1 if f.type.startswith("anomaly") else 0 for f in frames]
    colors = [{'anomaly_duplicate': 'orange', 'anomaly_insert': 'red', 'anomaly_discontinuity': 'purple'}.get(f.type, 'green') for f in frames]
    plt.figure(figsize=(15, 4)); plt.vlines(idxs, [0], labels, colors=colors, lw=2); plt.ylim(-0.1, 1.1)
    plt.xlabel("Indeks Bingkai"); plt.ylabel("Anomali (1=ya, 0=tidak)"); plt.title("Peta Anomali Temporal")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], c='r', lw=4, label='Penyisipan'), Line2D([0], [0], c='orange', lw=4, label='Duplikasi'), Line2D([0], [0], c='purple', lw=4, label='Diskontinuitas'), Line2D([0], [0], c='g', lw=4, label='Asli')]
    plt.legend(handles=legend_elements, loc='upper right'); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def write_unified_report(report_data: VideoResult | ComparisonResult, out_pdf: Path):
    """
    FUNGSI UTAMA PELAPORAN BARU: Mampu membuat laporan untuk video tunggal atau perbandingan.
    """
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, topMargin=50, bottomMargin=50, leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()
    story = []

    # --- Bagian Judul dan Metadata ---
    if isinstance(report_data, ComparisonResult):
        story.append(Paragraph("Laporan Forensik Perbandingan Video", styles['h1']))
        suspect = report_data.suspect_result
        base = report_data.baseline_result
        
        # Tabel perbandingan metadata
        table_data = [
            ['<b>Atribut</b>', '<b>Video Referensi (Baseline)</b>', '<b>Video Bukti (Suspect)</b>'],
            ['Nama File', Paragraph(Path(base.video).name, styles['Code']), Paragraph(Path(suspect.video).name, styles['Code'])],
            ['SHA-256', Paragraph(base.preservation_hash_sha256, styles['Code']), Paragraph(suspect.preservation_hash_sha256, styles['Code'])],
            ['Total Bingkai', base.summary['total_frames'], suspect.summary['total_frames']],
            ['Total Anomali', base.summary['total_anomaly'], suspect.summary['total_anomaly']]
        ]
        t = Table(table_data, colWidths=[100, 200, 200])
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                               ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                               ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black), ('BOX', (0,0), (-1,-1), 0.25, colors.black)]))
        story.append(t)

    else: # Laporan untuk video tunggal
        story.append(Paragraph("Laporan Forensik Video Tunggal", styles['h1']))
        suspect = report_data
        story.append(Paragraph(f"<b>Video Bukti:</b> {Path(suspect.video).name}", styles['Normal']))
        story.append(Paragraph(f"<b>Hash Preservasi (SHA-256):</b> <code>{suspect.preservation_hash_sha256}</code>", styles['Normal']))

    story.append(Spacer(1, 24))

    # --- Peta Anomali & Ringkasan ---
    story.append(Paragraph("Ringkasan dan Peta Anomali", styles['h2']))
    story.append(Paragraph(f"Total Bingkai: {suspect.summary['total_frames']}", styles['Normal']))
    story.append(Paragraph(f"Total Anomali Terdeteksi: {suspect.summary['total_anomaly']} ({suspect.summary['pct_anomaly']}%)", styles['Normal']))
    if suspect.temporal_plot and Path(suspect.temporal_plot).exists():
        story.append(Spacer(1, 12))
        story.append(PlatypusImage(suspect.temporal_plot, width=480, height=128))
    story.append(Spacer(1, 24))

    # --- Analisis Lanjutan ---
    story.append(Paragraph("Analisis Forensik Lanjutan", styles['h2']))
    if suspect.advanced_analysis:
        for module, findings in suspect.advanced_analysis.items():
            story.append(Paragraph(f"<b>Modul: {module.replace('_', ' ').title()}</b>", styles['h3']))
            if isinstance(findings, dict):
                for key, value in findings.items():
                    story.append(Paragraph(f"  - <b>{key.replace('_', ' ').title()}:</b> <code>{value}</code>", styles['Normal']))
            else:
                story.append(Paragraph(f"  - Temuan: {findings}", styles['Normal']))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 24))

    # --- Detail Lokalisasi Anomali ---
    if suspect.localizations:
        story.append(Paragraph("Detail Lokalisasi Anomali", styles['h2']))
        for i, loc in enumerate(suspect.localizations):
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"<b>Peristiwa #{i+1}: {loc['event'].replace('anomaly_', '').upper()}</b>", styles['h3']))
            story.append(Paragraph(f"<b>Rentang:</b> Bingkai {loc['start_frame']}–{loc.get('end_frame', loc['start_frame'])} "
                                  f"({loc['start_ts']:.2f}s–{loc.get('end_ts', loc['start_ts']):.2f}s)", styles['Normal']))
            if 'evidence' in loc and loc['evidence']: story.append(Paragraph(f"<b>Bukti Teknis:</b> <code>{json.dumps(loc['evidence'])}</code>", styles['Normal']))
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Bukti Visual (sampel bingkai anomali):</b>", styles['Normal']))
            
            # PENINGKATAN: Menampilkan hingga 7 gambar dalam tabel
            img_table_data = []
            row = []
            for img_path_str in loc["images"]:
                try:
                    img = PlatypusImage(img_path_str, width=110, height=62)
                    row.append(img)
                    if len(row) == 4: # 4 gambar per baris
                        img_table_data.append(row)
                        row = []
                except Exception:
                    row.append(Paragraph("Gagal Muat", styles['Normal']))
            if row: img_table_data.append(row) # Tambahkan sisa baris
            
            if img_table_data:
                img_table = Table(img_table_data)
                img_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
                story.append(img_table)

    doc.build(story)

###############################################################################
# Pipeline & Fungsi Inti
###############################################################################
def build_localizations(frames: list[FrameInfo]) -> list[dict]:
    locs = []; current_event = None
    if not frames: return locs
    for f in frames:
        if f.type.startswith("anomaly"):
            if current_event and current_event["event"] == f.type and f.index == current_event["end_frame"] + 1:
                current_event["end_frame"] = f.index; current_event["end_ts"] = f.timestamp
                # PENINGKATAN: Mengumpulkan lebih banyak gambar bukti
                if len(current_event["images"]) < 7: current_event["images"].append(f.img_path)
            else:
                if current_event: locs.append(current_event)
                current_event = {"event": f.type,"start_frame": f.index,"end_frame": f.index,"start_ts": f.timestamp,"end_ts": f.timestamp,"images": [f.img_path],"evidence": f.evidence}
        elif current_event:
            locs.append(current_event); current_event = None
    if current_event: locs.append(current_event)
    return locs

def generate_result(video_path: Path, pres_hash: str, frames: list[FrameInfo], meta: dict, adv_analysis: dict, out_dir: Path) -> VideoResult:
    plot_path = out_dir / f"temporal_{video_path.stem}.png"; plot_temporal_anomalies(frames, plot_path)
    locs = build_localizations(frames); total_anom = sum(1 for f in frames if f.type.startswith("anomaly"))
    total_frames = len(frames); summary = {"total_frames": total_frames,"total_anomaly": total_anom,"pct_anomaly": round(total_anom * 100 / total_frames, 2) if total_frames > 0 else 0}
    return VideoResult(video=str(video_path), preservation_hash_sha256=pres_hash, metadata=meta, frames=frames, summary=summary, temporal_plot=str(plot_path), localizations=locs, advanced_analysis=adv_analysis)

def save_manifest(report_data: VideoResult | ComparisonResult, out_path: Path):
    js = {"report_generated_utc": datetime.utcnow().isoformat() + "Z"}
    if isinstance(report_data, ComparisonResult):
        js['analysis_type'] = 'Comparison'
        js['results'] = {
            'baseline': asdict(report_data.baseline_result),
            'suspect': asdict(report_data.suspect_result)
        }
    else:
        js['analysis_type'] = 'Single File'
        js['results'] = asdict(report_data)
    out_path.write_text(json.dumps(js, indent=2), encoding='utf-8')

###############################################################################
# Alur Kerja DFRWS Utama
###############################################################################
def parse_args():
    p = argparse.ArgumentParser(description="Sistem Forensik Video Modular dengan Analisis Multi-Level (DFRWS).", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("videos", nargs="+", help="Path ke video bukti. Jika --baseline digunakan, ini adalah video suspek."); p.add_argument("--baseline", help="Path ke video referensi (asli) untuk perbandingan.", default=None)
    p.add_argument("--out_dir", default="hasil_forensik_lanjutan", help="Direktori untuk menyimpan semua hasil."); p.add_argument("--fps", type=int, default=30, help="Frame rate untuk ekstraksi (turunkan jika video panjang).")
    p.add_argument("--no-cleanup", action="store_true", help="Jangan hapus direktori bingkai setelah selesai.")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs_to_cleanup = []
    
    # Kumpulkan semua video yang perlu diproses
    all_video_paths_str = args.videos
    if args.baseline and args.baseline not in all_video_paths_str: all_video_paths_str.insert(0, args.baseline)

    video_results = {}

    # Tahap DFRWS 1-4: Identifikasi, Preservasi, Koleksi, Pemeriksaan
    for vid_path_str in all_video_paths_str:
        vid_path = Path(vid_path_str).resolve()
        if not vid_path.exists(): print(f"\n{Icons.ERROR} File tidak ditemukan: '{vid_path_str}'.", file=sys.stderr); continue

        print(f"\n\n{'='*25} MEMPROSES BUKTI: {vid_path.name} {'='*25}")
        
        # 1. IDENTIFIKASI
        print_stage_banner("Identifikasi", Icons.IDENTIFICATION, f"Mengidentifikasi file video: {vid_path.name}")
        print(f"  -> Path Absolut: {vid_path}\n  -> Ukuran: {vid_path.stat().st_size / 1e6:.2f} MB")
        
        # 2. PRESERVASI
        print_stage_banner("Preservasi", Icons.PRESERVATION, "Menghitung hash SHA-256 untuk integritas.")
        preservation_hash = calculate_sha256(vid_path); print(f"  -> Hash SHA-256: {preservation_hash}")
        
        # 3. KOLEKSI
        print_stage_banner("Koleksi", Icons.COLLECTION, f"Mengekstrak metadata & bingkai pada {args.fps} FPS.")
        metadata = ffprobe_metadata(vid_path); frames_dir = out_dir / f"frames_{vid_path.stem}"
        frame_dirs_to_cleanup.append(frames_dir); num_frames = extract_frames_ffmpeg(vid_path, frames_dir, args.fps)
        print(f"  {Icons.SUCCESS} {num_frames} bingkai diekstrak ke '{frames_dir.resolve()}'")

        # 4. PEMERIKSAAN
        print_stage_banner("Pemeriksaan", Icons.EXAMINATION, "Menghitung perceptual hash & menjalankan modul lanjutan.")
        frame_files = sorted(frames_dir.glob("frame_*.jpg")); frames = []
        for idx, fpath in tqdm(enumerate(frame_files), total=len(frame_files), desc="    Memeriksa Bingkai", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            ts = idx / args.fps; h = compute_hash(fpath); frames.append(FrameInfo(index=idx, timestamp=ts, img_path=str(fpath), hash=h))
        
        # Menjalankan modul analisis lanjutan
        advanced_analysis = {
            "encoder_metadata": analyze_encoder_metadata(metadata),
            "prnu_analysis": analyze_prnu_placeholder(frames),
            "gan_deepfake_detection": analyze_gan_deepfake_placeholder(frames),
        }
        print(f"  {Icons.SUCCESS} Pemeriksaan {len(frames)} bingkai & analisis lanjutan selesai.")
        
        # Membuat objek hasil sementara (sebelum analisis temporal)
        video_results[vid_path_str] = {"path": vid_path, "preservation_hash": preservation_hash, "frames": frames, "metadata": metadata, "advanced_analysis": advanced_analysis}

    # Tahap 5: ANALISIS
    print_stage_banner("Analisis Temporal", Icons.ANALYSIS, "Menganalisis data untuk mendeteksi anomali temporal.")
    final_reports = []
    if args.baseline and args.baseline in video_results and len(args.videos) > 0:
        suspect_str = args.videos[0]; base_str = args.baseline
        print(f"  {Icons.INFO} Mode Analisis: Perbandingan Baseline.")
        print(f"  -> Baseline: {Path(base_str).name}\n  -> Suspect: {Path(suspect_str).name}")
        base_data = video_results[base_str]; sus_data = video_results[suspect_str]
        analyze_pairwise(base_data['frames'], sus_data['frames']) # Analisis perbandingan
        analyze_single(base_data['frames']) # Analisis internal baseline
        
        base_res = generate_result(base_data['path'], base_data['preservation_hash'], base_data['frames'], base_data['metadata'], base_data['advanced_analysis'], out_dir)
        sus_res = generate_result(sus_data['path'], sus_data['preservation_hash'], sus_data['frames'], sus_data['metadata'], sus_data['advanced_analysis'], out_dir)
        
        comp_result = ComparisonResult(baseline_result=base_res, suspect_result=sus_res)
        final_reports.append(comp_result)

    else:
        print(f"  {Icons.INFO} Mode Analisis: Mandiri (tanpa baseline).")
        for vid_str, data in video_results.items():
            print(f"  -> Menganalisis konsistensi internal '{Path(vid_str).name}'...")
            analyze_single(data['frames'])
            res = generate_result(data['path'], data['preservation_hash'], data['frames'], data['metadata'], data['advanced_analysis'], out_dir)
            final_reports.append(res)
    print(f"  {Icons.SUCCESS} Analisis temporal selesai.")

    # Tahap 6: PELAPORAN
    print_stage_banner("Pelaporan", Icons.REPORTING, "Menghasilkan laporan PDF terpadu dan manifest JSON.")
    if final_reports:
        manifest_path = out_dir / "manifest.json"
        # Perlu logika untuk menyimpan manifest tunggal/ganda
        if len(final_reports) == 1: save_manifest(final_reports[0], manifest_path)
        else: print(f"  {Icons.INFO} Beberapa laporan dihasilkan, manifest tidak digabung (fitur masa depan).")
        print(f"  -> {Icons.SUCCESS} Manifest JSON disimpan.")
        
        for i, report_data in enumerate(final_reports):
            if isinstance(report_data, ComparisonResult):
                pdf_name = f"laporan_perbandingan_{Path(report_data.suspect_result.video).stem}.pdf"
            else:
                pdf_name = f"laporan_tunggal_{Path(report_data.video).stem}.pdf"
            pdf_path = out_dir / pdf_name
            write_unified_report(report_data, pdf_path)
            print(f"  -> {Icons.SUCCESS} Laporan PDF '{pdf_path.name}' berhasil dibuat.")
    else: print(f"  {Icons.ERROR} Tidak ada hasil untuk dilaporkan.")

    if not args.no_cleanup:
        print("\n" + "-"*35 + " PEMBERSIHAN " + "-"*35)
        for frame_dir in frame_dirs_to_cleanup:
            if frame_dir.exists(): shutil.rmtree(frame_dir); print(f"  {Icons.SUCCESS} Direktori sementara '{frame_dir.resolve()}' dihapus.")
    
    print(f"\n{Icons.SUCCESS} PROSES FORENSIK LANJUTAN SELESAI.\n")

if __name__ == "__main__":
    main()