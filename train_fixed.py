import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.models import Decoder, VGGEncoder
from utils.utils import (
    ImageFolderDataset,
    adaptive_instance_normalization,
    calc_mean_std,
    get_transform,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an AdaIN style-transfer decoder")

    parser.add_argument("--content_dir", type=str, default="./content_data")
    parser.add_argument("--style_dir", type=str, default="./style_data")
    parser.add_argument("--vgg", type=str, default="./vgg_normalised.pth")
    parser.add_argument("--experiment", type=str, default="experiment1")

    parser.add_argument("--final_size", type=int, default=256)
    parser.add_argument("--content_size", type=int, default=256)
    parser.add_argument("--style_size", type=int, default=256)
    parser.add_argument("--crop", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)

    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--style_weight", type=float, default=5.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=2)

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--optimizer_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.log_interval < 1 or args.save_interval < 1:
        raise ValueError("--log_interval and --save_interval must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Validate required paths.
    for path, label in (
        (args.vgg, "VGG weights"),
        (args.content_dir, "Content directory"),
        (args.style_dir, "Style directory"),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "args.txt", "w", encoding="utf-8") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")

    # Datasets and DataLoaders.
    content_transform = get_transform(
        args.content_size, args.crop, args.final_size
    )
    style_transform = get_transform(
        args.style_size, args.crop, args.final_size
    )

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    if len(content_dataset) < args.batch_size:
        raise ValueError("Not enough content images for one full batch.")
    if len(style_dataset) < args.batch_size:
        raise ValueError("Not enough style images for one full batch.")

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    num_batches = min(len(content_dataloader), len(style_dataloader))
    print("Content dataset size:", len(content_dataset))
    print("Style dataset size:", len(style_dataset))
    print("Number of paired batches per epoch:", num_batches)

    # Initialize each model exactly once.
    encoder = VGGEncoder(args.vgg).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    decoder = Decoder().to(device)
    trainable_parameters = [
        parameter for parameter in decoder.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError(
            "Decoder has no trainable parameters. Check Decoder in utils/models.py."
        )

    optimizer = optim.Adam(trainable_parameters, lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch),
    )

    # Restore checkpoint states only when explicitly requested.
    if args.resume:
        if not args.decoder_path or not args.optimizer_path:
            raise ValueError(
                "When using --resume, provide both --decoder_path and --optimizer_path."
            )
        if not Path(args.decoder_path).is_file():
            raise FileNotFoundError(f"Decoder checkpoint not found: {args.decoder_path}")
        if not Path(args.optimizer_path).is_file():
            raise FileNotFoundError(f"Optimizer checkpoint not found: {args.optimizer_path}")

        decoder.load_state_dict(
            torch.load(args.decoder_path, map_location=device)
        )
        optimizer.load_state_dict(
            torch.load(args.optimizer_path, map_location=device)
        )
        print("Resumed decoder and optimizer checkpoints.")

    # Quick shape check before training.
    content_batch = next(iter(content_dataloader)).to(device)
    style_batch = next(iter(style_dataloader)).to(device)
    with torch.no_grad():
        content_features = encoder(content_batch)
        style_features = encoder(style_batch)
        test_target = adaptive_instance_normalization(
            content_features[-1], style_features[-1]
        )
        test_output = decoder(test_target)

    print("Content batch shape:", tuple(content_batch.shape))
    print("Style batch shape:", tuple(style_batch.shape))
    print("VGG feature shapes:", [tuple(x.shape) for x in content_features])
    print("Decoder test output shape:", tuple(test_output.shape))

    if test_output.shape != content_batch.shape:
        raise RuntimeError(
            f"Decoder output shape {tuple(test_output.shape)} does not match "
            f"input image shape {tuple(content_batch.shape)}."
        )

    mse_loss = torch.nn.MSELoss()
    print("Training...")

    for epoch in range(args.epochs):
        decoder.train()
        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # Frozen encoder: no parameter updates, but gradients must flow
            # through encoder(g) back to the decoder.
            with torch.no_grad():
                c_feats = encoder(content_batch)
                s_feats = encoder(style_batch)
                target = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            generated = decoder(target)
            generated_features = encoder(generated)

            content_loss = (
                mse_loss(generated_features[-1], target) * args.content_weight
            )

            style_loss = torch.zeros((), device=device)
            for generated_feat, style_feat in zip(generated_features, s_feats):
                generated_mean, generated_std = calc_mean_std(generated_feat)
                style_mean, style_std = calc_mean_std(style_feat)
                style_loss = style_loss + mse_loss(
                    generated_mean, style_mean
                ) + mse_loss(generated_std, style_std)

            style_loss = style_loss * args.style_weight
            loss = content_loss + style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_closs += content_loss.item()
            running_sloss += style_loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                content=f"{content_loss.item():.4f}",
                style=f"{style_loss.item():.4f}",
            )

        scheduler.step()

        running_loss /= num_batches
        running_closs /= num_batches
        running_sloss /= num_batches

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(
                f"Epoch {epoch + 1}: Loss={running_loss:.4f}, "
                f"Content Loss={running_closs:.4f}, "
                f"Style Loss={running_sloss:.4f}"
            )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            torch.save(
                decoder.state_dict(),
                save_dir / f"decoder_{epoch + 1}.pth",
            )
            torch.save(
                optimizer.state_dict(),
                save_dir / f"optimizer_{epoch + 1}.pth",
            )

            # Save content, style, and generated images for visual inspection.
            with torch.no_grad():
                preview = torch.cat(
                    [content_batch, style_batch, generated.detach()], dim=0
                )
                save_image(
                    preview.clamp(0, 1),
                    save_dir / f"output_{epoch + 1}.png",
                    nrow=args.batch_size,
                )
            print(f"Saved checkpoint and preview for epoch {epoch + 1}.")

    print(f"Training finished. Outputs saved in: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
