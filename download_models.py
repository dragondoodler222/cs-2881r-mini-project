#!/usr/bin/env python3
"""
Script to download LLaMA models from HuggingFace Hub for offline use.
This script handles interruptions gracefully and can resume downloads.
"""

import os
import sys
import argparse
from huggingface_hub import snapshot_download
from pathlib import Path

# Directory where models will be stored
MODELS_DIR = "./models"

# Models to download
MODELS = {
    "llama2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-hf": "meta-llama/Llama-2-13b-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-nemoguard-8b": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "qwen2.5-7B": "Qwen/Qwen2.5-7B",
}


def download_model(model_name, repo_id, models_dir):
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_name: Local name for the model
        repo_id: HuggingFace repository ID
        models_dir: Directory to store models
    """
    local_dir = os.path.join(models_dir, model_name)
    
    print(f"\n{'='*80}")
    print(f"Downloading {model_name} from {repo_id}")
    print(f"Saving to: {local_dir}")
    print(f"{'='*80}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"],  # Only download safetensors
        )
        print(f"\n✓ Successfully downloaded {model_name}\n")
        return True
    except KeyboardInterrupt:
        print(f"\n\n⚠ Download interrupted by user. You can resume by running this script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error downloading {model_name}: {e}")
        print(f"  You can retry by running this script again - downloads will resume from where they stopped.\n")
        return False


def main():
    """Main function to download all models."""
    parser = argparse.ArgumentParser(
        description="Download LLaMA models from HuggingFace Hub for offline use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python download_models.py

  # Download specific model non-interactively
  python download_models.py --model llama2-7b-chat-hf

  # Download multiple models
  python download_models.py --model llama2-7b-chat-hf --model llama2-7b-hf

  # Download all 7B models
  python download_models.py --all-7b

  # Download all models
  python download_models.py --all
        """
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=list(MODELS.keys()),
        help="Specify which model(s) to download (can be used multiple times)"
    )
    parser.add_argument(
        "--all-7b",
        action="store_true",
        help="Download all 7B models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--models-dir",
        default=MODELS_DIR,
        help=f"Directory to save models (default: {MODELS_DIR})"
    )
    
    args = parser.parse_args()
    
    # Create models directory
    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("LLaMA Model Downloader")
    print("="*80)
    print(f"\nModels will be saved to: {models_dir}")
    
    # Determine which models to download
    models_to_download = {}
    
    if args.all:
        models_to_download = MODELS
    elif args.all_7b:
        models_to_download = {
            "llama2-7b-hf": MODELS["llama2-7b-hf"],
            "llama2-7b-chat-hf": MODELS["llama2-7b-chat-hf"]
        }
    elif args.model:
        for model_name in args.model:
            models_to_download[model_name] = MODELS[model_name]
    else:
        # Interactive mode
        print("\nThis script will download the following models:")
        for name, repo in MODELS.items():
            print(f"  - {name}: {repo}")
        
        print("\nNote: These are large models (7B: ~13GB, 13B: ~25GB each)")
        print("      Downloads can be interrupted and resumed.\n")
        
        # Ask for confirmation
        response = input("Do you want to proceed? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Download cancelled.")
            sys.exit(0)
        
        # Ask which models to download
        print("\nWhich models would you like to download?")
        print("1. llama2-7b-chat-hf only (recommended for testing)")
        print("2. All 7B models (7b-hf and 7b-chat-hf)")
        print("3. All models (7B and 13B)")
        print("4. Custom selection")
        
        choice = input("\nEnter your choice [1-4]: ").strip()
        
        if choice == "1":
            models_to_download = {"llama2-7b-chat-hf": MODELS["llama2-7b-chat-hf"]}
        elif choice == "2":
            models_to_download = {
                "llama2-7b-hf": MODELS["llama2-7b-hf"],
                "llama2-7b-chat-hf": MODELS["llama2-7b-chat-hf"]
            }
        elif choice == "3":
            models_to_download = MODELS
        elif choice == "4":
            models_to_download = {}
            for name in MODELS.keys():
                response = input(f"Download {name}? [y/N]: ").strip().lower()
                if response in ['y', 'yes']:
                    models_to_download[name] = MODELS[name]
        else:
            print("Invalid choice. Downloading llama2-7b-chat-hf only.")
            models_to_download = {"llama2-7b-chat-hf": MODELS["llama2-7b-chat-hf"]}
    
    if not models_to_download:
        print("No models selected. Exiting.")
        sys.exit(0)
    
    # Download selected models
    success_count = 0
    failed_models = []
    
    print(f"\nDownloading {len(models_to_download)} model(s)...\n")
    
    for model_name, repo_id in models_to_download.items():
        success = download_model(model_name, repo_id, models_dir)
        if success:
            success_count += 1
        else:
            failed_models.append(model_name)
    
    # Print summary
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"\nSuccessfully downloaded: {success_count}/{len(models_to_download)} models")
    
    if failed_models:
        print(f"\nFailed downloads: {', '.join(failed_models)}")
        print("You can retry by running this script again.")
    else:
        print("\n✓ All models downloaded successfully!")
        print(f"\nModels are stored in: {models_dir}")
        print("\nTo use these models, the code has been configured to use local paths.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

