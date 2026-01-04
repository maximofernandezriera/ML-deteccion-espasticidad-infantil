#!/usr/bin/env python3
"""
Script para convertir los archivos NPZ del dataset a videos MP4 visualizables.
Autor: Máximo Fernández Riera
Fecha: Enero 2026
"""

import numpy as np
import cv2
from pathlib import Path

def main():
    # Rutas
    project_root = Path(__file__).parent
    data_file = project_root / 'data' / 'raw' / 'data_100_50_50.npz'
    labels_file = project_root / 'data' / 'raw' / 'target_100_50_50.npz'
    output_dir = project_root / 'videos_exportados'
    
    # Verificar que existe el dataset
    if not data_file.exists():
        print(f"❌ No se encontró el archivo: {data_file}")
        return
    
    # Cargar dataset
    print("📂 Cargando dataset NPZ...")
    data = np.load(data_file, allow_pickle=True)
    videos = data['arr_0']
    print(f"   ✓ Videos cargados: {videos.shape}")
    print(f"   ✓ Formato: {videos.shape[0]} videos × {videos.shape[1]} frames × {videos.shape[2]}×{videos.shape[3]} px × {videos.shape[4]} canales")
    
    # Cargar etiquetas si existen
    labels = None
    if labels_file.exists():
        labels_data = np.load(labels_file, allow_pickle=True)
        labels = labels_data['arr_0']
        print(f"   ✓ Etiquetas cargadas: {labels.shape}")
        print(f"   ✓ Clases: {np.unique(labels)}")
    
    # Crear directorio de salida
    output_dir.mkdir(exist_ok=True)
    
    # Exportar solo una muestra representativa (2 videos por clase)
    print(f"\n🎬 Exportando videos de muestra a: {output_dir}/")
    
    videos_exportados = 0
    videos_por_clase = {}
    
    # Agrupar por clase
    if labels is not None:
        for clase in np.unique(labels):
            indices = np.where(labels == clase)[0][:2]  # 2 por clase
            videos_por_clase[int(clase)] = indices
    else:
        # Si no hay etiquetas, exportar los primeros 16
        videos_por_clase[0] = list(range(min(16, len(videos))))
    
    # Exportar videos
    for clase, indices in videos_por_clase.items():
        for idx in indices:
            video_data = videos[idx]
            
            # Nombre del archivo
            if labels is not None:
                output_path = output_dir / f'clase{clase}_video{idx:03d}.mp4'
            else:
                output_path = output_dir / f'video_{idx:03d}.mp4'
            
            # Configurar codec y writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (50, 50))
            
            # Escribir frames
            for frame in video_data:
                # Convertir RGB a BGR (OpenCV usa BGR)
                frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            videos_exportados += 1
            print(f"   ✓ Exportado: {output_path.name}")
    
    print(f"\n✅ {videos_exportados} videos exportados exitosamente")
    print(f"📁 Ubicación: {output_dir.absolute()}")
    print(f"\n💡 Los videos tienen resolución 50×50 px (dataset preprocesado para ML)")
    print(f"   Para verlos mejor, amplía el reproductor de video.")

if __name__ == "__main__":
    main()
