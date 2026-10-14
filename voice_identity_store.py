"""Encrypted, consent-gated voice profiles for the research prototype.

A match is only a candidate signal. The receptionist must obtain a spoken
full-name check before attaching any patient context or revealing a name.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class VoiceIdentityStore:
    MATCH_THRESHOLD = float(os.getenv("VOICE_IDENTITY_MATCH_THRESHOLD", "0.78"))
    MINIMUM_MARGIN = float(os.getenv("VOICE_IDENTITY_MINIMUM_MARGIN", "0.08"))

    def __init__(self, path: str | Path, encryption_key: str | bytes):
        self.path = Path(path)
        if isinstance(encryption_key, str):
            encryption_key = encryption_key.encode("ascii")
        self.cipher = Fernet(encryption_key)

    @staticmethod
    def normalize_embedding(embedding) -> list[float]:
        if hasattr(embedding, "detach"):
            embedding = embedding.detach()
        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu()
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        while isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
            embedding = embedding[0]
        values = [float(value) for value in embedding]
        magnitude = math.sqrt(sum(value * value for value in values))
        if not values or magnitude <= 0:
            raise ValueError("Voice embedding is empty or invalid.")
        return [value / magnitude for value in values]

    @staticmethod
    def similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return -1.0
        return sum(a * b for a, b in zip(left, right))

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "profiles": []}
        encrypted = self.path.read_bytes()
        try:
            payload = self.cipher.decrypt(encrypted)
        except InvalidToken as error:
            raise RuntimeError("Voice profile file could not be decrypted.") from error
        data = json.loads(payload.decode("utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("profiles"), list):
            raise RuntimeError("Voice profile file format is invalid.")
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        temporary.write_bytes(self.cipher.encrypt(payload))
        temporary.replace(self.path)

    def enroll(
        self,
        patient_name: str,
        embedding,
        *,
        explicit_consent: bool,
        secondary_identity_verified: bool,
    ) -> dict:
        clean_name = " ".join(str(patient_name or "").split())
        if not explicit_consent:
            raise PermissionError("Explicit voice-profile consent is required.")
        if not secondary_identity_verified:
            raise PermissionError("Secondary identity verification is required.")
        if len(clean_name) < 2:
            raise ValueError("A patient name is required.")

        normalized = self.normalize_embedding(embedding)
        data = self._load()
        profile = {
            "profile_id": str(uuid.uuid4()),
            "patient_name": clean_name,
            "embedding": normalized,
            "consented_at": datetime.now(timezone.utc).isoformat(),
            "revoked_at": None,
        }
        data["profiles"].append(profile)
        self._save(data)
        return {"profile_id": profile["profile_id"], "patient_name": clean_name}

    def match(self, embedding) -> dict:
        query = self.normalize_embedding(embedding)
        active = [
            profile for profile in self._load()["profiles"]
            if not profile.get("revoked_at")
        ]
        if not active:
            return {"matched": False, "reason": "no_active_profiles"}

        scored = sorted(
            (
                self.similarity(query, profile["embedding"]),
                profile,
            )
            for profile in active
        )
        best_score, best_profile = scored[-1]
        second_score = scored[-2][0] if len(scored) > 1 else -1.0
        margin = best_score - second_score
        if best_score < self.MATCH_THRESHOLD:
            return {
                "matched": False,
                "reason": "below_threshold",
                "similarity": round(best_score, 4),
            }
        if len(scored) > 1 and margin < self.MINIMUM_MARGIN:
            return {
                "matched": False,
                "reason": "ambiguous_candidates",
                "similarity": round(best_score, 4),
                "margin": round(margin, 4),
            }
        return {
            "matched": True,
            "candidate_profile_id": best_profile["profile_id"],
            "candidate_patient_name": best_profile["patient_name"],
            "similarity": round(best_score, 4),
            "margin": round(margin, 4),
            "identity_verified": False,
            "instruction": "Require a spoken full-name check before using patient context.",
        }

    def revoke(self, profile_id: str, *, explicit_confirmation: bool) -> bool:
        if not explicit_confirmation:
            raise PermissionError("Explicit revocation confirmation is required.")
        data = self._load()
        changed = False
        for profile in data["profiles"]:
            if profile.get("profile_id") == profile_id and not profile.get("revoked_at"):
                profile["revoked_at"] = datetime.now(timezone.utc).isoformat()
                changed = True
        if changed:
            self._save(data)
        return changed
