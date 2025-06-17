# Rencana Pengembangan Sistem Forensik Video

Dokumen ini menjelaskan rencana singkat pengembangan modul tambahan pada skrip `ForensikVideo.py`.

## Tujuan
- Menambah deteksi manipulasi temporal (insertion, deletion, duplication) dengan beberapa teknik (hash, optical flow, SSIM, analisis audio).
- Meningkatkan fitur ekspor hasil ke format CSV selain PDF dan JSON.

## Langkah Implementasi
1. **Penambahan Library**: memanfaatkan `scikit-image` untuk SSIM dan `scipy` untuk pembacaan audio.
2. **Fungsi Baru**
   - `extract_audio_wav` untuk mengekstrak audio dari video.
   - `analyze_audio_discontinuity` menghitung z-score perubahan amplitudo audio.
   - `export_anomaly_csv` menghasilkan file CSV berisi detail frame anomali.
3. **Penyempurnaan Analisis**
   - `analyze_single` kini menghitung optical flow dan SSIM pada setiap pasangan frame.
   - `analyze_pairwise` menandai frame yang dihapus dari baseline.
4. **Pipeline Utama**
   - Audio diekstrak pada tahap pemeriksaan dan dianalisis.
   - Hasil analisis diekspor ke PDF, JSON, dan CSV.
5. **Pembersihan**
   - File audio sementara dan direktori frame dihapus otomatis kecuali opsi `--no-cleanup` dipakai.

## Cara Menjalankan
1. Pastikan dependensi telah terpasang (`opencv-python-headless`, `scikit-image`, `imagehash`, `tqdm`, `reportlab`, `matplotlib`, `scipy`).
2. Jalankan skrip dengan:
   ```bash
   python3 ForensikVideo.py path_video.mp4 --out_dir hasil
   ```
3. Jika memiliki video baseline:
   ```bash
   python3 ForensikVideo.py suspect.mp4 --baseline original.mp4 --out_dir hasil
   ```
4. Laporan PDF dan file CSV akan tersedia di direktori `hasil`.
