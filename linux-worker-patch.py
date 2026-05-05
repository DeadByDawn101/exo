#!/usr/bin/env python3
"""Star Platinum — Linux Worker Patch. Rips out election on Linux."""
import os, sys, shutil

EXO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(EXO_ROOT, "src", "exo")

def patch_election():
    path = os.path.join(SRC, "shared", "election.py")
    with open(path) as f: content = f.read()
    old = '                await self._campaign(candidates, campaign_timeout=0.0)'
    new = '''                import sys as _sys
                if _sys.platform == "linux":
                    logger.info("Linux worker: skipping self-election, waiting for Mac Master...")
                    await anyio.sleep(30)
                    logger.info("Linux worker: ready to accept Mac Master")
                else:
                    await self._campaign(candidates, campaign_timeout=0.0)'''
    if old in content:
        content = content.replace(old, new)
        # Also stop Linux from cancelling other campaigns
        old_cancel = '            logger.info("Cancelling other campaign")'
        new_cancel = '''            import sys as _sys2
            if _sys2.platform == "linux":
                logger.info("Linux worker: accepting peer election (not competing)")
                if self._campaign_cancel_scope is not None:
                    self._campaign_cancel_scope.cancel()
                if self._campaign_done is not None:
                    await self._campaign_done.wait()
                await self.elect(msg)
                continue
            logger.info("Cancelling other campaign")'''
        content = content.replace(old_cancel, new_cancel, 1)
        with open(path, 'w') as f: f.write(content)
        print("[OK] election.py patched")
    else:
        print("[WARN] election.py — target not found")

def patch_main():
    path = os.path.join(SRC, "main.py")
    with open(path) as f: content = f.read()
    old = 'seniority=1_000_000 if args.force_master else 0,'
    new = 'seniority=1_000_000 if args.force_master else 0,\n            is_candidate=not (__import__("sys").platform == "linux" and not args.force_master),'
    if old in content and 'is_candidate' not in content:
        content = content.replace(old, new)
        with open(path, 'w') as f: f.write(content)
        print("[OK] main.py patched")

def patch_bootstrap():
    path = os.path.join(SRC, "worker", "runner", "bootstrap.py")
    with open(path) as f: content = f.read()
    if '_should_use_tinygrad' in content:
        print("[SKIP] bootstrap.py already patched")
        return
    func = '\ndef _should_use_tinygrad() -> bool:\n    import sys\n    if sys.platform != "linux": return False\n    try:\n        from tinygrad import Device\n        return Device.DEFAULT in ("NV", "CUDA", "GPU")\n    except ImportError: return False\n\n'
    content = content.replace('def entrypoint(', func + 'def entrypoint(')
    old_mlx = '''        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender,
                cancel_receiver=cancel_receiver,
            )'''
    new_mlx = '''        elif _should_use_tinygrad():
            from exo.worker.engines.tinygrad.builder import TinygradBuilder
            logger.info("Using TinygradBuilder (NVIDIA GPU detected)")
            builder = TinygradBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender,
                cancel_receiver=cancel_receiver,
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches
            apply_mlx_patches()
            from exo.worker.engines.mlx.builder import MlxBuilder
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender,
                cancel_receiver=cancel_receiver,
            )'''
    content = content.replace(old_mlx, new_mlx)
    with open(path, 'w') as f: f.write(content)
    print("[OK] bootstrap.py patched")

def patch_mlx_imports():
    files = [
        os.path.join(SRC, "worker", "runner", "llm_inference", "batch_generator.py"),
        os.path.join(SRC, "worker", "runner", "llm_inference", "model_output_parsers.py"),
    ]
    replacements = [
        ("import mlx.core as mx\n", "try:\n    import mlx.core as mx\nexcept ImportError:\n    mx = None\n"),
        ("from mlx_lm.tokenizer_utils import TokenizerWrapper\n", "try:\n    from mlx_lm.tokenizer_utils import TokenizerWrapper\nexcept ImportError:\n    TokenizerWrapper = None\n"),
        ("from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model\n", "try:\n    from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model\nexcept ImportError:\n    DeepseekV4Model = None\n"),
        ("from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model\n", "try:\n    from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model\nexcept ImportError:\n    DeepseekV32Model = None\n"),
        ("from mlx_lm.models.gpt_oss import Model as GptOssModel\n", "try:\n    from mlx_lm.models.gpt_oss import Model as GptOssModel\nexcept ImportError:\n    GptOssModel = None\n"),
    ]
    for fp in files:
        if not os.path.exists(fp): continue
        with open(fp) as f: content = f.read()
        changed = False
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                changed = True
        if changed:
            with open(fp, 'w') as f: f.write(content)
            print(f"[OK] {os.path.basename(fp)} — MLX imports guarded")

print("=" * 50)
print(" Star Platinum — Linux Worker Patch")
print("=" * 50)
os.system(f"cd {EXO_ROOT} && git checkout -- src/ 2>/dev/null")
print("[OK] Reset to clean state")
patch_election()
patch_main()
patch_bootstrap()
patch_mlx_imports()
print("\n[DONE] All patches applied. Start with:")
print('  EXO_LIBP2P_NAMESPACE=1.0.71 uv run python -m exo --libp2p-port 30000 --bootstrap-peers "/ip4/192.168.1.247/tcp/<PORT>"')
