"""Create (or update) the CreditForge Hugging Face Docker Space and upload source.

HF rebuilds the Docker image from the repo's `Dockerfile` on their infra, so we
just upload the source tree + a Space README carrying the required frontmatter.

Usage:
    huggingface-cli login            # once, paste a write token
    python deploy/push_to_hf.py      # uses default space name <user>/CreditForge
    python deploy/push_to_hf.py my-space-name
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ROOT = Path(__file__).resolve().parent.parent

# Source actually needed to build + run the Docker image on HF.
ALLOW = [
    "Dockerfile", ".dockerignore", "requirements.txt", "requirements-serve.txt",
    "creditforge/**", "app/api/**", "app/__init__.py", "app/dashboard/**",
]
# Never upload these (regenerable / heavy / local).
IGNORE = [
    ".git/**", ".venv/**", "mlruns/**", "**/__pycache__/**", "**/*.pyc",
    "app/dashboard/node_modules/**", "app/dashboard/.next/**", "app/dashboard/out/**",
    "creditforge/data/**", "creditforge/artifacts/**", "creditforge/reports/*.html",
    ".github/**",
]


def main() -> int:
    api = HfApi()
    user = whoami()["name"]
    space_name = sys.argv[1] if len(sys.argv) > 1 else "CreditForge"
    repo_id = f"{user}/{space_name}"
    print(f"Target Space: {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    exist_ok=True)

    # Upload the source tree (HF will docker-build it).
    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=str(REPO_ROOT),
        allow_patterns=ALLOW, ignore_patterns=IGNORE,
        commit_message="Deploy CreditForge single-container app",
    )
    # Overwrite README.md with the Space frontmatter version (sdk/app_port).
    api.upload_file(
        repo_id=repo_id, repo_type="space",
        path_or_fileobj=str(REPO_ROOT / "deploy" / "hf" / "README.md"),
        path_in_repo="README.md",
        commit_message="Space README frontmatter",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n✅ Uploaded. HF is now building the Docker image.")
    print(f"   Watch the build + open the app: {url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # surface auth/errors cleanly
        print(f"\n✗ {type(e).__name__}: {e}")
        print("If this is an auth error, run:  huggingface-cli login  (write token)")
        raise SystemExit(1)
