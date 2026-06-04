#!/usr/bin/env python3
"""Headless CLI for Pixel Shuffler."""

from pixel_shuffler import PixelShufflerTrainer, TrainingConfig


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pixel Shuffler — neural pixel rearrangement")
    parser.add_argument("content", help="Content image path (structure)")
    parser.add_argument("style", help="Style image path (appearance)")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    config = TrainingConfig(
        iterations=args.iterations,
        image_size=args.image_size,
        learning_rate=args.lr,
    )
    trainer = PixelShufflerTrainer(config)

    def on_progress(step: int, total: int, losses: dict, _preview) -> None:
        print(f"[{step}/{total}] content={losses['content']:.3f} style={losses['style']:.3f} tv={losses['tv']:.3f}")

    trainer.train(args.content, args.style, on_progress=on_progress)
    print("Done. See output/final_stylized.png")


if __name__ == "__main__":
    main()
