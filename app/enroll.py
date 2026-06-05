import argparse
import sys
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

from app import config
from app.face_db import FaceDB


def _load_face_app() -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        root="models",
        providers=config.get_ort_providers(),
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage face data for face recognition"
    )
    parser.add_argument("--photo", help="Path to face photo (.jpg/.png)")
    parser.add_argument("--name", help="Name of the person to enroll")
    parser.add_argument("--list", action="store_true", help="List all enrolled faces")
    parser.add_argument("--delete", type=int, metavar="ID", help="Delete face by ID")
    args = parser.parse_args()

    db = FaceDB()

    if args.list:
        rows = db.list_all()
        if not rows:
            print("No faces enrolled yet.")
        else:
            print(f"\n{'ID':<5}  {'Name':<30}  {'Enrolled'}")
            print("-" * 58)
            for id_, name, created_at in rows:
                print(f"{id_:<5}  {name:<30}  {created_at}")
            print()
        return

    if args.delete is not None:
        if db.delete(args.delete):
            print(f"[OK] Face ID {args.delete} deleted successfully")
        else:
            print(f"[ERROR] ID {args.delete} not found")
        return

    if not args.photo or not args.name:
        parser.print_help()
        print("\n[ERROR] Use --photo and --name to enroll a face")
        sys.exit(1)

    photo_path = Path(args.photo)
    if not photo_path.exists():
        print(f"[ERROR] File not found: {args.photo}")
        sys.exit(1)

    img = cv2.imread(str(photo_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {args.photo}")
        sys.exit(1)

    print("[INFO] Loading InsightFace buffalo_l ...")
    face_app = _load_face_app()
    faces = face_app.get(img)

    if not faces:
        print("[ERROR] No face detected in the photo")
        print("        Make sure the photo is clear, well-lit, and the face is facing forward")
        sys.exit(1)

    if len(faces) > 1:
        print(f"[WARN] {len(faces)} faces detected — using the largest face")
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

    face_id = db.enroll(args.name, faces[0].embedding)
    print(f"[OK] '{args.name}' enrolled successfully (ID: {face_id})")


if __name__ == "__main__":
    main()
