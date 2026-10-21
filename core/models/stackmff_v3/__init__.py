"""StackMFF V3: multi-focus image stack fusion by pixel-wise classification over the input images.

network.py  - StackMFFLoop (final model: single_scale=True, linear head) and StackMFFV3 (earlier multi-stage variant)
model_io.py - FINAL_ARGS, build(), load_checkpoint() for the released weights, load() for training runs
Dataloader.py - pre-synthesized stacks (validation; training without on-the-fly synthesis)
"""
__version__ = '1.0'
