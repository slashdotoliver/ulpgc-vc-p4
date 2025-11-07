#!/usr/bin/env python3
"""
Convierte JSONs de LabelMe + imágenes (jpg/jpeg) a estructura YOLO:
 output/
   images/
   labels/

Cada imagen se copia a output/images/ y cada .txt (mismo nombre base) se guarda en output/labels/

Formato por defecto (YOLO normalized):
 <class_id> <x_center> <y_center> <width> <height>

Opciones:
 --use-names : escribir el nombre de la clase en lugar del id (ej. 'plate' ...)
 --pixels    : escribir coordenadas en píxeles (x_center y_center width height)
"""
import os
import json
import glob
import argparse
import shutil
import base64
import cv2
import numpy as np

def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return x_min, y_min, x_max, y_max

def load_image_from_json_or_disk(json_path, imagePath):
    json_dir = os.path.dirname(json_path)
    if imagePath:
        candidate = os.path.join(json_dir, imagePath)
        if os.path.exists(candidate):
            img = cv2.imread(candidate)
            return img, candidate
        # maybe imagePath is absolute
        if os.path.exists(imagePath):
            img = cv2.imread(imagePath)
            return img, imagePath

    base = os.path.splitext(os.path.basename(json_path))[0]
    for ext in (".jpg", ".jpeg", ".png"):
        cand = os.path.join(json_dir, base + ext)
        if os.path.exists(cand):
            img = cv2.imread(cand)
            return img, cand

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_data = data.get("imageData", None)
    if img_data:
        try:
            img_bytes = base64.b64decode(img_data)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            # save temporary image path next to json
            temp_path = os.path.join(json_dir, base + "_fromjson.jpg")
            cv2.imwrite(temp_path, img)
            return img, temp_path
        except Exception:
            return None, None

    return None, None

def process_json(json_path, out_images_dir, out_labels_dir, class_name_to_id, use_names=False, pixels=False, copy_images=True):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    imagePath = data.get("imagePath", None)
    img, img_path_used = load_image_from_json_or_disk(json_path, imagePath)
    if img is None:
        raise RuntimeError(f"No se encontró ni la imagen ni imageData para {json_path}")

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(img_path_used))[0]
    out_image_path = os.path.join(out_images_dir, os.path.basename(img_path_used))
    out_label_path = os.path.join(out_labels_dir, base_name + ".txt")

    if copy_images:
        shutil.copy2(img_path_used, out_image_path)

    lines = []
    shapes = data.get("shapes", [])
    for shape in shapes:
        label = shape.get("label", "").strip()
        points = shape.get("points", [])
        if not points:
            continue
        x_min, y_min, x_max, y_max = bbox_from_points(points)
        x_c = (x_min + x_max) / 2.0
        y_c = (y_min + y_max) / 2.0
        bw = (x_max - x_min)
        bh = (y_max - y_min)

        if label not in class_name_to_id:
            # ignorar etiquetas que no están en la lista de clases
            continue
        cls_id = class_name_to_id[label]

        if pixels:
            # center and size in pixels (integers)
            if use_names:
                first = label
            else:
                first = str(cls_id)
            line = f"{first} {int(x_c)} {int(y_c)} {int(bw)} {int(bh)}"
        else:
            # normalized floats
            x_c_n = x_c / w
            y_c_n = y_c / h
            bw_n = bw / w
            bh_n = bh / h
            if use_names:
                first = label
            else:
                first = str(cls_id)
            line = f"{first} {x_c_n:.6f} {y_c_n:.6f} {bw_n:.6f} {bh_n:.6f}"

        lines.append(line)

    with open(out_label_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
        else:
            f.write("")

    return out_image_path, out_label_path, len(lines)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_images_dir = os.path.join(args.out_dir, "images")
    out_labels_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    class_name_to_id = {name: i for i, name in enumerate(args.classes)}

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print("No se encontraron .json en", args.input_dir)
        return

    total_boxes = 0
    processed = 0
    for j in json_files:
        try:
            img_out, lbl_out, n = process_json(
                j,
                out_images_dir,
                out_labels_dir,
                class_name_to_id,
                use_names=args.use_names,
                pixels=args.pixels,
                copy_images=not args.no_copy_images
            )
            processed += 1
            total_boxes += n
            if args.verbose:
                print(f"[OK] {os.path.basename(j)} -> {os.path.basename(img_out)}, {n} cajas -> {os.path.basename(lbl_out)}")
        except Exception as e:
            print(f"[ERROR] {os.path.basename(j)}: {e}")

    print(f"Procesados: {processed} JSONs. Total cajas exportadas: {total_boxes}")
    print("Salida en:", args.out_dir)
    print("Images:", out_images_dir)
    print("Labels:", out_labels_dir)
    print("Mapping clases (name -> id):", class_name_to_id)
    if args.use_names:
        print("Formato: <class_name> <x> <y> <w> <h>")
    elif args.pixels:
        print("Formato: <class_id> <x_center_px> <y_center_px> <w_px> <h_px>")
    else:
        print("Formato (YOLO normalized): <class_id> <x_center> <y_center> <w> <h>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LabelMe JSONs to YOLO txt & arrange into images/ labels/")
    parser.add_argument("--input-dir", "-i", required=True, help="Directorio con los .json y las imágenes")
    parser.add_argument("--out-dir", "-o", required=True, help="Directorio de salida (crea images/ y labels/)")
    parser.add_argument("--classes", "-c", nargs="+", default=["plate"], help="Lista de clases; orden define class_id (ej: -c plate)")
    parser.add_argument("--use-names", action="store_true", help="En lugar de usar id numérico, escribe el nombre de la clase como primer campo")
    parser.add_argument("--pixels", action="store_true", help="Generar coordenadas en píxeles (x_center y_center w h) en lugar de normalizado")
    parser.add_argument("--no-copy-images", action="store_true", help="No copiar imágenes a out_dir/images (solo generar labels)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs verbosos")
    args = parser.parse_args()
    main(args)
