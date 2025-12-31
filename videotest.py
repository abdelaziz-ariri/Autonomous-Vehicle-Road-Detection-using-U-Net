#!/usr/bin/env python3
"""
Script d'inférence vidéo en temps réel avec triple affichage
Affiche 3 fenêtres simultanément pendant le traitement :
1. Vidéo originale
2. Masque de segmentation
3. Overlay (masque transparent)

Usage:
    python realtime_segmentation.py --model models/best_model.h5 --video video.mp4
    
Options:
    --model : Chemin vers le modèle (.h5 ou SavedModel)
    --video : Chemin vers la vidéo
    --input-size : Taille d'entrée du modèle (défaut: 224)
    --save : Sauvegarder la vidéo de sortie (défaut: False)
    --output : Chemin de sortie si --save activé
    --fps-limit : Limiter les FPS pour ralentir l'affichage (optionnel)
"""

import os
import argparse
import time
import numpy as np
import cv2
import tensorflow as tf


# juste après `import tensorflow as tf`
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU(s) détecté(s) :", gpus)
    # Activer memory growth pour éviter allocation complète de la VRAM
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("⚠️ Erreur memory growth:", e)
else:
    print("⚠️ Aucun GPU détecté par TensorFlow. L'inférence se fera sur CPU.")



# ===================================================================
# PALETTE DE COULEURS POUR LA SEGMENTATION
# ===================================================================

COLOR_DICT = {
    0: (128, 128, 128),  # Sky
    1: (128, 0, 0),      # Building
    2: (192, 192, 128),  # Pole
    3: (128, 64, 128),   # Road
    4: (0, 0, 192),      # Sidewalk / Pavement
    5: (128, 128, 0),    # Tree
    6: (192, 128, 128),  # SignSymbol
    7: (64, 64, 128),    # Fence
    8: (64, 0, 128),     # Car
    9: (64, 64, 0),      # Pedestrian
    10: (0, 128, 192),   # Bicyclist
    255: (0, 0, 0)
}

CLASS_NAMES = {
    0: 'Sky', 1: 'Building', 2: 'Pole', 3: 'Road',
    4: 'Sidewalk', 5: 'Tree', 6: 'Sign', 7: 'Fence',
    8: 'Car', 9: 'Pedestrian', 10: 'Bicyclist'
}

# ===================================================================
# FONCTIONS DE CHARGEMENT ET TRAITEMENT
# ===================================================================

def load_model_safe(path):
    """Charge le modèle avec gestion des erreurs."""
    print(f"🔄 Chargement du modèle: {path}")
    
    if os.path.isdir(path):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except:
            return tf.saved_model.load(path)
    else:
        try:
            # Essai avec safe_mode=False pour les modèles legacy
            return tf.keras.models.load_model(path, compile=False, safe_mode=False)
        except:
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                print(f"❌ Erreur: {e}")
                raise

def preprocess_frame(frame, input_size=(224, 224)):
    """Prétraite une frame pour l'inférence."""
    img = cv2.resize(frame, input_size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return img

def predict_mask(model, frame, input_size=(224, 224)):
    """Prédit le masque de segmentation."""
    img = preprocess_frame(frame, input_size)
    pred = model.predict(np.expand_dims(img, 0), verbose=0)
    mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
    return mask

def mask_to_color(mask):
    """Convertit un masque en image couleur."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, col in COLOR_DICT.items():
        if k == 255:
            continue
        out[mask == k] = col
    return out

def add_info_overlay(frame, text, position=(10, 30), 
                     font_scale=0.7, color=(0, 255, 255), thickness=2):
    """Ajoute du texte sur une frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return frame

def create_legend_image():
    """Crée une image de légende pour les classes."""
    legend_height = 30
    legend_width = 250
    n_classes = len(CLASS_NAMES)
    
    legend = np.zeros((n_classes * legend_height, legend_width, 3), dtype=np.uint8)
    
    for idx, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
        y_start = idx * legend_height
        y_end = (idx + 1) * legend_height
        color = COLOR_DICT[class_id]
        
        # Rectangle de couleur
        cv2.rectangle(legend, (0, y_start), (60, y_end), color, -1)
        
        # Bordure
        cv2.rectangle(legend, (0, y_start), (60, y_end), (255, 255, 255), 1)
        
        # Texte
        cv2.putText(legend, class_name, (70, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return legend

# ===================================================================
# FONCTION PRINCIPALE - AFFICHAGE TRIPLE VIEW EN TEMPS RÉEL
# ===================================================================

def run_realtime_triple_view(model, video_path, input_size=(224, 224),
                             save_output=False, output_path=None,
                             fps_limit=None, window_scale=0.8):
    """
    Traite et affiche la vidéo en temps réel avec 3 fenêtres.
    
    Args:
        model: Modèle de segmentation chargé
        video_path: Chemin vers la vidéo
        input_size: Taille d'entrée du modèle
        save_output: Sauvegarder la vidéo (True/False)
        output_path: Chemin de sortie si save_output=True
        fps_limit: Limiter les FPS pour ralentir (None = vitesse max)
        window_scale: Échelle d'affichage des fenêtres (0.5 = moitié)
    """
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Impossible d'ouvrir: {video_path}")
    
    # Propriétés de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculer les dimensions d'affichage
    display_width = int(width * window_scale)
    display_height = int(height * window_scale)
    
    print(f"📹 Vidéo: {width}x{height} @ {fps:.1f} FPS")
    print(f"📊 Total frames: {total_frames}")
    print(f"🖥️  Affichage: {display_width}x{display_height}")
    print(f"\n⚡ Traitement en temps réel - Appuyez sur 'q' pour quitter\n")
    
    # Créer les fenêtres
    cv2.namedWindow('1. ORIGINAL', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2. SEGMENTATION', cv2.WINDOW_NORMAL)
    cv2.namedWindow('3. OVERLAY', cv2.WINDOW_NORMAL)
    cv2.namedWindow('LEGEND', cv2.WINDOW_NORMAL)
    
    # Positionner les fenêtres (ajuster selon ton écran)
    spacing = 20
    cv2.resizeWindow('1. ORIGINAL', display_width, display_height)
    cv2.resizeWindow('2. SEGMENTATION', display_width, display_height)
    cv2.resizeWindow('3. OVERLAY', display_width, display_height)
    
    cv2.moveWindow('1. ORIGINAL', 0, 0)
    cv2.moveWindow('2. SEGMENTATION', display_width + spacing, 0)
    cv2.moveWindow('3. OVERLAY', (display_width + spacing) * 2, 0)
    cv2.moveWindow('LEGEND', 0, display_height + 50)
    
    # Afficher la légende
    legend = create_legend_image()
    cv2.imshow('LEGEND', legend)
    
    # VideoWriter si sauvegarde activée
    out = None
    if save_output and output_path:
        # Créer une vidéo avec les 3 vues côte à côte
        margin = 10
        output_width = width * 3 + margin * 4
        output_height = height + margin * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        print(f"💾 Sauvegarde activée: {output_path}")
    
    # Variables de timing
    frame_idx = 0
    start_time = time.time()
    processing_times = []
    
    # Boucle principale
    try:
        while True:
            frame_start = time.time()
            
            # Lire la frame
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Fin de la vidéo")
                break
            
            # Prédiction
            pred_start = time.time()
            mask = predict_mask(model, frame, input_size=input_size)
            pred_time = time.time() - pred_start
            
            # Redimensionner le masque
            mask_resized = cv2.resize(mask, (width, height), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Créer le masque coloré
            mask_colored = mask_to_color(mask_resized)
            
            # Créer l'overlay
            overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
            
            # Ajouter les informations sur chaque vue
            progress = f"Frame {frame_idx + 1}/{total_frames} ({(frame_idx + 1)/total_frames*100:.1f}%)"
            fps_display = f"FPS: {1.0/pred_time:.1f} | Inference: {pred_time*1000:.1f}ms"
            
            frame_display = frame.copy()
            mask_display = mask_colored.copy()
            overlay_display = overlay.copy()
            
            add_info_overlay(frame_display, "ORIGINAL", (10, 30), 
                           font_scale=1.0, thickness=2)
            add_info_overlay(frame_display, progress, (10, 60), 
                           font_scale=0.6, thickness=1)
            
            add_info_overlay(mask_display, "SEGMENTATION", (10, 30),
                           font_scale=1.0, thickness=2)
            add_info_overlay(mask_display, fps_display, (10, 60),
                           font_scale=0.6, thickness=1)
            
            add_info_overlay(overlay_display, "OVERLAY", (10, 30),
                           font_scale=1.0, thickness=2)
            
            # Afficher les 3 fenêtres
            cv2.imshow('1. ORIGINAL', frame_display)
            cv2.imshow('2. SEGMENTATION', mask_display)
            cv2.imshow('3. OVERLAY', overlay_display)
            
            # Sauvegarder si activé
            if out is not None:
                canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                margin = 10
                x_positions = [margin, width + margin * 2, width * 2 + margin * 3]
                
                canvas[margin:margin+height, x_positions[0]:x_positions[0]+width] = frame
                canvas[margin:margin+height, x_positions[1]:x_positions[1]+width] = mask_colored
                canvas[margin:margin+height, x_positions[2]:x_positions[2]+width] = overlay
                
                out.write(canvas)
            
            # Statistiques
            frame_time = time.time() - frame_start
            processing_times.append(pred_time)
            
            # Afficher les stats toutes les 30 frames
            if (frame_idx + 1) % 30 == 0:
                avg_time = np.mean(processing_times[-30:])
                avg_fps = 1.0 / avg_time
                elapsed = time.time() - start_time
                eta = (total_frames - frame_idx - 1) * avg_time
                
                print(f"Frame {frame_idx + 1:4d}/{total_frames} | "
                      f"FPS: {avg_fps:5.1f} | "
                      f"Inference: {avg_time*1000:5.1f}ms | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"ETA: {eta:.1f}s")
            
            # Gestion du clavier
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️  Arrêt demandé par l'utilisateur")
                break
            elif key == ord('p'):
                print("⏸️  Pause - Appuyez sur n'importe quelle touche pour continuer")
                cv2.waitKey(0)
            elif key == ord('s'):
                # Sauvegarder la frame actuelle
                cv2.imwrite(f'frame_{frame_idx}_original.png', frame)
                cv2.imwrite(f'frame_{frame_idx}_mask.png', mask_colored)
                cv2.imwrite(f'frame_{frame_idx}_overlay.png', overlay)
                print(f"📸 Frame {frame_idx} sauvegardée")
            
            # Limiter les FPS si demandé
            if fps_limit:
                target_time = 1.0 / fps_limit
                elapsed = time.time() - frame_start
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Interruption par Ctrl+C")
    
    finally:
        # Libérer les ressources
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Statistiques finales
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        avg_inference = np.mean(processing_times) if processing_times else 0
        
        print(f"\n{'='*60}")
        print(f"📊 STATISTIQUES FINALES")
        print(f"{'='*60}")
        print(f"Frames traitées : {frame_idx}/{total_frames}")
        print(f"Temps total     : {total_time:.2f}s")
        print(f"FPS moyen       : {avg_fps:.2f}")
        print(f"Inférence moy.  : {avg_inference*1000:.2f}ms")
        print(f"{'='*60}")
        
        if out is not None:
            print(f"💾 Vidéo sauvegardée: {output_path}")

# ===================================================================
# FONCTION ALTERNATIVE - VUE UNIQUE COMBINÉE
# ===================================================================

def run_realtime_single_view(model, video_path, input_size=(224, 224),
                             save_output=False, output_path=None):
    """
    Alternative : Affiche les 3 vues dans une seule fenêtre.
    Utile si tu as un petit écran.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Impossible d'ouvrir: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Mode fenêtre unique: {width}x{height} @ {fps:.1f} FPS")
    print(f"⚡ Appuyez sur 'q' pour quitter\n")
    
    cv2.namedWindow('Triple View Segmentation', cv2.WINDOW_NORMAL)
    
    out = None
    if save_output and output_path:
        margin = 10
        output_width = width * 3 + margin * 4
        output_height = height + margin * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    frame_idx = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            mask = predict_mask(model, frame, input_size=input_size)
            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_colored = mask_to_color(mask_resized)
            overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
            
            # Créer le canvas
            margin = 10
            canvas_width = width * 3 + margin * 4
            canvas_height = height + margin * 2
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            x_positions = [margin, width + margin * 2, width * 2 + margin * 3]
            
            canvas[margin:margin+height, x_positions[0]:x_positions[0]+width] = frame
            canvas[margin:margin+height, x_positions[1]:x_positions[1]+width] = mask_colored
            canvas[margin:margin+height, x_positions[2]:x_positions[2]+width] = overlay
            
            # Ajouter les titres
            add_info_overlay(canvas, 'ORIGINAL', (x_positions[0] + 10, 30), 
                           font_scale=0.8, thickness=2)
            add_info_overlay(canvas, 'SEGMENTATION', (x_positions[1] + 10, 30),
                           font_scale=0.8, thickness=2)
            add_info_overlay(canvas, 'OVERLAY', (x_positions[2] + 10, 30),
                           font_scale=0.8, thickness=2)
            
            # Info globale
            progress = f'Frame: {frame_idx + 1}/{total_frames}'
            add_info_overlay(canvas, progress, (canvas_width // 2 - 100, canvas_height - 15),
                           font_scale=0.7, thickness=2)
            
            cv2.imshow('Triple View Segmentation', canvas)
            
            if out is not None:
                out.write(canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Interruption")
    
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"\n✅ Traité: {frame_idx} frames en {total_time:.2f}s ({frame_idx/total_time:.2f} FPS)")

# ===================================================================
# PROGRAMME PRINCIPAL
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inférence vidéo en temps réel avec segmentation sémantique'
    )
    parser.add_argument('--model', required=True, 
                       help='Chemin vers le modèle (.h5 ou SavedModel)')
    parser.add_argument('--video', required=True,
                       help='Chemin vers la vidéo')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Taille d\'entrée du modèle (défaut: 224)')
    parser.add_argument('--save', action='store_true',
                       help='Sauvegarder la vidéo de sortie')
    parser.add_argument('--output', default='output_triple_view.mp4',
                       help='Chemin de sortie (si --save activé)')
    parser.add_argument('--fps-limit', type=int, default=None,
                       help='Limiter les FPS d\'affichage (optionnel)')
    parser.add_argument('--window-scale', type=float, default=0.8,
                       help='Échelle d\'affichage des fenêtres (défaut: 0.8)')
    parser.add_argument('--single-window', action='store_true',
                       help='Afficher les 3 vues dans une seule fenêtre')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎬 REAL-TIME SEMANTIC SEGMENTATION - TRIPLE VIEW")
    print("="*60)
    
    # Charger le modèle
    print(f"\n1️⃣ Chargement du modèle...")
    model = load_model_safe(args.model)
    print(f"✅ Modèle chargé")
    
    # Vérifier la vidéo
    if not os.path.exists(args.video):
        print(f"❌ Vidéo introuvable: {args.video}")
        return
    
    print(f"✅ Vidéo: {args.video}")
    
    # Lancer le traitement
    print(f"\n2️⃣ Démarrage du traitement en temps réel...")
    print(f"\n💡 Contrôles:")
    print(f"   q : Quitter")
    print(f"   p : Pause")
    print(f"   s : Sauvegarder la frame actuelle")
    print()
    
    input_size = (args.input_size, args.input_size)
    
    if args.single_window:
        run_realtime_single_view(
            model=model,
            video_path=args.video,
            input_size=input_size,
            save_output=args.save,
            output_path=args.output if args.save else None
        )
    else:
        run_realtime_triple_view(
            model=model,
            video_path=args.video,
            input_size=input_size,
            save_output=args.save,
            output_path=args.output if args.save else None,
            fps_limit=args.fps_limit,
            window_scale=args.window_scale
        )
    
    print("\n✅ Terminé! 🎉")

if __name__ == '__main__':
    main()