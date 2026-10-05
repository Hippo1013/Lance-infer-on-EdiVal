"""Create a durable visual review of completed smoke conversations."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    width, cell_height = 448, 540
    sheet = Image.new("RGB", (width * (1 + max(len(x[1]) for x in run["samples"])), cell_height * len(run["samples"])), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    sections = []
    for row, (session, instructions) in enumerate(run["samples"]):
        cards = []
        for turn in range(len(instructions) + 1):
            name = "turn_0_input.png" if turn == 0 else f"turn_{turn}.png"
            file = args.run / session / name
            with Image.open(file) as image:
                image = image.convert("RGB")
                preview = ImageOps.contain(image, (width - 16, 336))
            x, y = turn * width, row * cell_height
            draw.text((x + 8, y + 8), f"{session.split('/')[0].upper()} | " + ("Input" if turn == 0 else f"Turn {turn}"),
                      fill="black", font=font)
            sheet.paste(preview, (x + (width - preview.width) // 2, y + 36))
            text = "Original source image" if turn == 0 else instructions[turn - 1]
            lines, current = [], ""
            for word in text.replace("—", "-").split():
                candidate = f"{current} {word}".strip()
                if draw.textlength(candidate, font=font) > width - 24:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
            draw.multiline_text((x + 10, y + 384), "\n".join(lines), fill="black", font=font, spacing=3)
            cards.append(f'<article><h3>{"Input" if turn == 0 else f"Turn {turn}"}</h3>'
                         f'<img src="{session}/{name}"><p>{html.escape(text)}</p></article>')
        sections.append(f'<section><h2>{html.escape(session)}</h2><div class="row">{"".join(cards)}</div></section>')
    sheet.save(args.run / "review.png")
    document = '<!doctype html><meta charset="utf-8"><title>Multi-turn smoke review</title>' \
        '<style>body{font:16px system-ui;margin:24px}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}' \
        'img{width:100%;object-fit:contain}p{line-height:1.5}article{min-width:0}' \
        '@media(max-width:900px){.row{grid-template-columns:repeat(2,1fr)}}</style>' \
        '<h1>Original input and generated turns</h1>' + "".join(sections)
    (args.run / "review.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
