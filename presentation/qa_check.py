"""Geometry and text-fit QA for the generated deck (no renderer available).

Flags the defects that are actually user-visible: shapes off-slide, shapes
inside the 0.5" margin, overlapping text boxes, and text likely to overflow
its container.
"""
import sys
from pptx import Presentation
from pptx.util import Emu

SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.5
# Rough per-character width as a fraction of font size, for common fonts.
CHAR_W = 0.50
LINE_H = 1.22


def inches(v):
    return Emu(v).inches if v is not None else None


def text_of(shape):
    if not shape.has_text_frame:
        return ""
    return "\n".join(p.text for p in shape.text_frame.paragraphs)


def max_font(shape):
    sizes = []
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            if r.font.size:
                sizes.append(r.font.size.pt)
    return max(sizes) if sizes else 12.0


def estimate_height(text, font_pt, box_w_in):
    """Estimate rendered height in inches for wrapped text."""
    if not text.strip():
        return 0.0
    char_w_in = (font_pt * CHAR_W) / 72.0
    usable = max(box_w_in - 0.10, 0.2)
    chars_per_line = max(int(usable / char_w_in), 1)
    lines = 0
    for para in text.split("\n"):
        lines += max(1, -(-len(para) // chars_per_line))
    return lines * (font_pt * LINE_H) / 72.0


prs = Presentation(sys.argv[1] if len(sys.argv) > 1 else "Multi_Agent_Fact_Checker_PPE.pptx")
problems = []

for idx, slide in enumerate(prs.slides, start=1):
    boxes = []
    for shape in slide.shapes:
        x, y = inches(shape.left), inches(shape.top)
        w, h = inches(shape.width), inches(shape.height)
        if None in (x, y, w, h):
            continue
        name = shape.shape_type
        label = (text_of(shape)[:34].replace("\n", " ") or str(name))

        # 1. off-slide
        if x < -0.02 or y < -0.02 or x + w > SLIDE_W + 0.02 or y + h > SLIDE_H + 0.02:
            # decorative circles are deliberately bled off-canvas
            if shape.has_text_frame and text_of(shape).strip():
                problems.append(f"S{idx}: OFF-SLIDE  '{label}'  ({x:.2f},{y:.2f} {w:.2f}x{h:.2f})")

        # 2. text inside the page margin
        if shape.has_text_frame and text_of(shape).strip():
            if x < MARGIN - 0.02 or x + w > SLIDE_W - MARGIN + 0.02:
                problems.append(f"S{idx}: MARGIN    '{label}'  x={x:.2f} right={x+w:.2f}")
            if y < 0.25 or y + h > SLIDE_H - 0.20:
                problems.append(f"S{idx}: MARGIN-Y  '{label}'  y={y:.2f} bottom={y+h:.2f}")

            # 3. text overflow estimate
            fs = max_font(shape)
            need = estimate_height(text_of(shape), fs, w)
            if need > h + 0.06:
                problems.append(
                    f"S{idx}: OVERFLOW? '{label}'  needs~{need:.2f}in has {h:.2f}in (font {fs:.0f}pt, w {w:.2f})"
                )
            boxes.append((x, y, w, h, label))

    # 4. overlapping text boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay, aw, ah, al = boxes[i]
            bx, by, bw, bh, bl = boxes[j]
            ox = min(ax + aw, bx + bw) - max(ax, bx)
            oy = min(ay + ah, by + bh) - max(ay, by)
            if ox > 0.12 and oy > 0.12:
                problems.append(f"S{idx}: OVERLAP   '{al}'  <->  '{bl}'  ({ox:.2f}x{oy:.2f}in)")

print(f"Slides: {len(prs.slides)}   Issues: {len(problems)}\n")
for p in problems:
    print(" ", p)
if not problems:
    print("  No geometry or text-fit issues detected.")
