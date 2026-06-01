"""PAGE XML Viewer — overlay PAGE XML annotations on a manuscript image.

Usage:
    python page_viewer.py                           # open via file dialogs
    python page_viewer.py annotation.xml            # load XML, prompt for image
    python page_viewer.py image.jpg annotation.xml  # pre-load both files
"""

import argparse
import json
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Pillow is required: pip install pillow")

# Common PAGE XML namespace URIs — namespace is auto-detected at parse time,
# but this list is kept for documentation.
_PAGE_NS_PATTERN = re.compile(
    r"http://schema\.primaresearch\.org/PAGE/gts/pagecontent/\d{4}-\d{2}-\d{2}"
)

# Layer definitions: key → (fill_color, outline_color, display_label, outline_width)
LAYERS = {
    "regions":   ("#4488FF", "#2255CC", "Regions",   2),
    "lines":     ("#44DD44", "#22AA22", "Lines",     2),
    "words":     ("#FF44FF", "#CC22CC", "Words",     2),
    "baselines": ("#FF8800", "#CC5500", "Baselines", 2),
    "glyphs":    ("#44DDDD", "#22AAAA", "Glyphs",   1),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ANNOTATION_EXTENSIONS = {".xml", ".json"}


def _point_in_polygon(px: float, py: float, flat_pts: list[float]) -> bool:
    """Ray-casting point-in-polygon. flat_pts = [x0,y0, x1,y1, ...]."""
    n = len(flat_pts) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = flat_pts[2 * i],     flat_pts[2 * i + 1]
        xj, yj = flat_pts[2 * j],     flat_pts[2 * j + 1]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _point_near_polyline(px: float, py: float, flat_pts: list[float],
                         threshold: float = 6.0) -> bool:
    """True if (px, py) is within threshold pixels of any segment."""
    n = len(flat_pts) // 2
    thresh_sq = threshold ** 2
    for i in range(n - 1):
        x1, y1 = flat_pts[2 * i],     flat_pts[2 * i + 1]
        x2, y2 = flat_pts[2 * i + 2], flat_pts[2 * i + 3]
        dx, dy = x2 - x1, y2 - y1
        seg_sq = dx * dx + dy * dy
        if seg_sq == 0:
            dist_sq = (px - x1) ** 2 + (py - y1) ** 2
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_sq))
            dist_sq = (px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2
        if dist_sq <= thresh_sq:
            return True
    return False


# ---------------------------------------------------------------------------
# PAGE XML parser
# ---------------------------------------------------------------------------

def _parse_points(points_str: str) -> list[tuple[int, int]]:
    """Parse a PAGE XML points attribute into a list of (x, y) int tuples."""
    result = []
    for token in points_str.strip().split():
        x, y = token.split(",")
        result.append((int(float(x)), int(float(y))))
    return result


def _text_equiv(element, ns: str) -> str:
    """Return the Unicode text content of an element's TextEquiv child, or ''."""
    te = element.find(f"{{{ns}}}TextEquiv/{{{ns}}}Unicode")
    return te.text.strip() if te is not None and te.text else ""


def _coords(element, ns: str) -> list[tuple[int, int]]:
    """Return the Coords points of a PAGE XML element, or []."""
    c = element.find(f"{{{ns}}}Coords")
    if c is None:
        return []
    pts = c.get("points", "")
    return _parse_points(pts) if pts.strip() else []


def parse_page_xml(xml_path: str) -> dict:
    """
    Parse a PAGE XML file and return a dict with keys:
        image_filename, image_width, image_height,
        regions, lines, words, baselines, glyphs
    Each annotation is a dict with: id, coords, text, attrs, parent_id (where relevant).
    baselines have: id, coords, parent_id only.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Auto-detect namespace from root tag, e.g. {http://…}PcGts
    m = re.match(r"\{(.+?)\}", root.tag)
    ns = m.group(1) if m else ""

    page_el = root.find(f"{{{ns}}}Page") if ns else root.find("Page")
    if page_el is None:
        raise ValueError("No <Page> element found in PAGE XML.")

    result = {
        "image_filename": page_el.get("imageFilename", ""),
        "image_width":    int(page_el.get("imageWidth", 0)),
        "image_height":   int(page_el.get("imageHeight", 0)),
        "regions":   [],
        "lines":     [],
        "words":     [],
        "baselines": [],
        "glyphs":    [],
    }

    for region_el in page_el.findall(f"{{{ns}}}TextRegion") if ns else page_el.findall("TextRegion"):
        region_id = region_el.get("id", "")
        region = {
            "id":        region_id,
            "type":      region_el.get("type", "TextRegion"),
            "coords":    _coords(region_el, ns),
            "text":      _text_equiv(region_el, ns),
            "attrs":     {k: v for k, v in region_el.attrib.items() if k not in ("id", "type")},
            "parent_id": None,
            "layer":     "regions",
        }
        result["regions"].append(region)

        for line_el in region_el.findall(f"{{{ns}}}TextLine") if ns else region_el.findall("TextLine"):
            line_id = line_el.get("id", "")
            line = {
                "id":        line_id,
                "type":      "TextLine",
                "coords":    _coords(line_el, ns),
                "text":      _text_equiv(line_el, ns),
                "attrs":     {k: v for k, v in line_el.attrib.items() if k != "id"},
                "parent_id": region_id,
                "layer":     "lines",
            }
            result["lines"].append(line)

            # Baseline is a polyline child of TextLine
            bl_el = line_el.find(f"{{{ns}}}Baseline") if ns else line_el.find("Baseline")
            if bl_el is not None:
                pts = _parse_points(bl_el.get("points", "")) if bl_el.get("points", "").strip() else []
                if pts:
                    result["baselines"].append({
                        "id":        f"{line_id}_baseline",
                        "type":      "Baseline",
                        "coords":    pts,
                        "text":      "",
                        "attrs":     {},
                        "parent_id": line_id,
                        "layer":     "baselines",
                    })

            for word_el in line_el.findall(f"{{{ns}}}Word") if ns else line_el.findall("Word"):
                word_id = word_el.get("id", "")
                word = {
                    "id":        word_id,
                    "type":      "Word",
                    "coords":    _coords(word_el, ns),
                    "text":      _text_equiv(word_el, ns),
                    "attrs":     {k: v for k, v in word_el.attrib.items() if k != "id"},
                    "parent_id": line_id,
                    "layer":     "words",
                }
                result["words"].append(word)

                for glyph_el in word_el.findall(f"{{{ns}}}Glyph") if ns else word_el.findall("Glyph"):
                    result["glyphs"].append({
                        "id":        glyph_el.get("id", ""),
                        "type":      "Glyph",
                        "coords":    _coords(glyph_el, ns),
                        "text":      _text_equiv(glyph_el, ns),
                        "attrs":     {k: v for k, v in glyph_el.attrib.items() if k != "id"},
                        "parent_id": word_id,
                        "layer":     "glyphs",
                    })

    return result


def parse_kraken_json(path: str) -> dict:
    """Parse Kraken's native JSON segmentation format (type: 'baselines')."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") not in ("baselines", "bbox"):
        raise ValueError(f"Unrecognised Kraken JSON type: {data.get('type')!r}")

    result = {
        "image_filename": data.get("imagename", ""),
        "image_width":    0,
        "image_height":   0,
        "regions":   [],
        "lines":     [],
        "words":     [],
        "baselines": [],
        "glyphs":    [],
    }

    # regions is {type_name: [region_obj, ...], ...}
    for rtype, rlist in data.get("regions", {}).items():
        for rdata in rlist:
            boundary = rdata.get("boundary") or []
            result["regions"].append({
                "id":        rdata.get("id", ""),
                "type":      rtype,
                "coords":    [tuple(p) for p in boundary],
                "text":      "",
                "attrs":     {},
                "parent_id": None,
                "layer":     "regions",
            })

    for line in data.get("lines", []):
        line_id = line.get("id", "")
        boundary = line.get("boundary") or []
        baseline = line.get("baseline") or []
        parent_region = (line.get("regions") or [None])[0]

        result["lines"].append({
            "id":        line_id,
            "type":      "TextLine",
            "coords":    [tuple(p) for p in boundary],
            "text":      line.get("text") or "",
            "attrs":     {"tags": str(line.get("tags", ""))},
            "parent_id": parent_region,
            "layer":     "lines",
        })

        if baseline:
            result["baselines"].append({
                "id":        f"{line_id}_baseline",
                "type":      "Baseline",
                "coords":    [tuple(p) for p in baseline],
                "text":      "",
                "attrs":     {},
                "parent_id": line_id,
                "layer":     "baselines",
            })

    return result


def load_annotation_file(path: str) -> tuple[dict, str]:
    """
    Load a PAGE XML or Kraken JSON annotation file.
    Returns (data_dict, format_label) where format_label is 'PAGE XML' or 'Kraken JSON'.
    Auto-detects format by peeking at the first non-whitespace byte.
    """
    with open(path, "rb") as f:
        first = f.read(512).lstrip()
    if first.startswith(b"{"):
        return parse_kraken_json(path), "Kraken JSON"
    else:
        return parse_page_xml(path), "PAGE XML"


def _resolve_image(image_filename: str, xml_path: str) -> Path | None:
    """Try several strategies to locate the image referenced in the PAGE XML."""
    xml_dir = Path(xml_path).parent
    candidates = [
        xml_dir / image_filename,
        xml_dir / Path(image_filename).name,
    ]
    if Path(image_filename).is_absolute():
        candidates.insert(0, Path(image_filename))
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class PageViewerApp(tk.Tk):
    def __init__(self, preload_xml: str | None = None, preload_image: str | None = None):
        super().__init__()
        self.title("PAGE XML Viewer")
        self.geometry("1280x800")
        self.minsize(600, 400)

        # Application state
        self.page_data: dict | None = None
        self.xml_path: str | None = None
        self.pil_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None  # keep reference to prevent GC

        self.scale: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self._drag_start: tuple[int, int] | None = None
        self._drag_moved: bool = False

        # Canvas item → annotation dict
        self.item_data: dict[int, dict] = {}
        # Canvas item → (normal_outline, normal_width) for restoring after hover/select
        self.item_style: dict[int, tuple[str, int]] = {}

        self._hovered_item: int | None = None
        self._selected_item: int | None = None

        # Layer visibility toggles
        self.layer_visible: dict[str, tk.BooleanVar] = {
            k: tk.BooleanVar(value=True) for k in LAYERS
        }

        self._build_ui()
        self._bind_events()

        # Pre-load files if given on the command line (deferred until window renders)
        if preload_xml or preload_image:
            self.after(100, lambda: self._preload(preload_xml, preload_image))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Toolbar ────────────────────────────────────────────────────
        toolbar = ttk.Frame(self, padding=(6, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open XML…",   command=self.open_xml).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open Image…", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(toolbar, text="Zoom In",     command=lambda: self._zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Zoom Out",    command=lambda: self._zoom(1 / 1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reset View",  command=self.reset_view).pack(side=tk.LEFT, padx=2)

        self._status_var = tk.StringVar(value="Open a PAGE XML file to begin.")
        ttk.Label(toolbar, textvariable=self._status_var, foreground="#555555").pack(
            side=tk.LEFT, padx=12
        )

        # ── Main paned area ────────────────────────────────────────────
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: layer toggles
        left = ttk.Frame(paned, padding=10)
        paned.add(left, weight=0)

        ttk.Label(left, text="Layers", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 6))
        for key, (fill, outline, label, _) in LAYERS.items():
            row = ttk.Frame(left)
            row.pack(anchor=tk.W, pady=1)
            # Color swatch
            swatch = tk.Canvas(row, width=14, height=14, highlightthickness=0, bg=self.cget("bg"))
            swatch.create_rectangle(1, 1, 13, 13, fill=fill, outline=outline)
            swatch.pack(side=tk.LEFT, padx=(0, 4))
            ttk.Checkbutton(
                row, text=label,
                variable=self.layer_visible[key],
                command=self._redraw,
            ).pack(side=tk.LEFT)

        # Center: canvas
        canvas_frame = ttk.Frame(paned)
        paned.add(canvas_frame, weight=3)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right: properties sidebar
        right = ttk.Frame(paned, padding=10)
        paned.add(right, weight=0)

        ttk.Label(right, text="Properties", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 6))
        self._sidebar = tk.Text(
            right, width=26, state=tk.DISABLED, wrap=tk.WORD,
            font=("Menlo", 10), relief=tk.FLAT, bg=self.cget("bg"),
        )
        self._sidebar.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Event bindings
    # ------------------------------------------------------------------

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>",       self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>",          self._on_motion)
        # Zoom: macOS/Windows uses MouseWheel; Linux uses Button-4/5
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>",   self._on_scroll)
        self.canvas.bind("<Button-5>",   self._on_scroll)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def open_xml(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Annotation File",
            filetypes=[("Annotation files", "*.xml *.json"), ("All files", "*.*")],
        )
        if path:
            self._load_xml(path)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_image(path)

    def _load_xml(self, path: str) -> None:
        try:
            self.page_data, fmt = load_annotation_file(path)
            self.xml_path = path
        except Exception as exc:
            messagebox.showerror("Parse error", f"Could not parse annotation file:\n{exc}")
            return

        counts = (
            f"{len(self.page_data['regions'])} regions, "
            f"{len(self.page_data['lines'])} lines, "
            f"{len(self.page_data['words'])} words"
        )
        self._status_var.set(f"{Path(path).name} [{fmt}] — {counts}")

        # Try to locate the image referenced in the XML
        img_filename = self.page_data.get("image_filename", "")
        if img_filename:
            img_path = _resolve_image(img_filename, path)
            if img_path:
                self._load_image(str(img_path))
                return
            # Image not found — ask the user
            if messagebox.askyesno(
                "Image not found",
                f"Could not find:\n  {img_filename}\n\nLocate the image manually?",
            ):
                self.open_image()
                return

        self._redraw()

    def _load_image(self, path: str) -> None:
        try:
            self.pil_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image error", f"Could not open image:\n{exc}")
            return
        self.reset_view()

    def _preload(self, xml_path: str | None, image_path: str | None) -> None:
        if image_path:
            self._load_image(image_path)
        if xml_path:
            self._load_xml(xml_path)

    # ------------------------------------------------------------------
    # View / zoom / pan
    # ------------------------------------------------------------------

    def reset_view(self) -> None:
        if self.pil_image is None:
            return
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        iw, ih = self.pil_image.size
        self.scale = min(cw / iw, ch / ih, 1.0)
        self.pan_x = (cw - iw * self.scale) / 2
        self.pan_y = (ch - ih * self.scale) / 2
        self._redraw()

    def _zoom(self, factor: float, cx: float | None = None, cy: float | None = None) -> None:
        if cx is None:
            cx = self.canvas.winfo_width() / 2
        if cy is None:
            cy = self.canvas.winfo_height() / 2
        old = self.scale
        self.scale = max(0.05, min(20.0, self.scale * factor))
        ratio = self.scale / old
        self.pan_x = cx - (cx - self.pan_x) * ratio
        self.pan_y = cy - (cy - self.pan_y) * ratio
        self._redraw()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self.pil_image is not None and self.scale == 1.0:
            self.reset_view()
        else:
            self._redraw()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_moved = False

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        if abs(dx) > 3 or abs(dy) > 3:
            self._drag_moved = True
        self.pan_x += dx
        self.pan_y += dy
        self._drag_start = (event.x, event.y)
        self._redraw()

    def _on_release(self, event: tk.Event) -> None:
        if not self._drag_moved:
            self._try_select(event.x, event.y)
        self._drag_start = None
        self._drag_moved = False

    def _on_scroll(self, event: tk.Event) -> None:
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            factor = 1.15
        else:
            factor = 1 / 1.15
        self._zoom(factor, event.x, event.y)

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag_moved:
            return
        self._update_hover(event.x, event.y)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.pan_x, y * self.scale + self.pan_y

    def _scale_points(self, coords: list[tuple[int, int]]) -> list[float]:
        flat = []
        for x, y in coords:
            cx, cy = self._to_canvas(x, y)
            flat.extend([cx, cy])
        return flat

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self.item_data.clear()
        self.item_style.clear()
        self._hovered_item = None
        self._selected_item = None

        if self.pil_image:
            iw, ih = self.pil_image.size
            new_w = max(1, int(iw * self.scale))
            new_h = max(1, int(ih * self.scale))
            resized = self.pil_image.resize((new_w, new_h), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(resized)
            self.canvas.create_image(int(self.pan_x), int(self.pan_y),
                                     anchor=tk.NW, image=self.photo)

        if self.page_data:
            self._draw_layers()

    def _draw_layers(self) -> None:
        data = self.page_data

        # Draw from broadest to narrowest so narrower items sit on top
        for layer_key in ("regions", "lines", "words", "baselines", "glyphs"):
            if not self.layer_visible[layer_key].get():
                continue
            fill, outline, _, width = LAYERS[layer_key]
            anns = data[layer_key]
            for ann in anns:
                pts = self._scale_points(ann["coords"])
                if len(pts) < 4:
                    continue
                if layer_key == "baselines":
                    item = self.canvas.create_line(
                        pts, fill=outline, width=width + 1,
                        tags=(layer_key, "annotation"),
                        capstyle=tk.ROUND, joinstyle=tk.ROUND,
                    )
                else:
                    item = self.canvas.create_polygon(
                        pts, fill="",
                        outline=outline, width=width,
                        tags=(layer_key, "annotation"),
                    )
                self.item_data[item] = {**ann, "_pts": pts}
                self.item_style[item] = (outline, width)

    # ------------------------------------------------------------------
    # Hover & selection
    # ------------------------------------------------------------------

    def _ann_item_at(self, cx: int, cy: int) -> int | None:
        """Return the topmost annotation at canvas position (cx, cy), or None."""
        for item in reversed(list(self.item_data)):
            ann = self.item_data[item]
            pts = ann.get("_pts", [])
            if not pts:
                continue
            if ann.get("layer") == "baselines":
                if _point_near_polyline(cx, cy, pts):
                    return item
            else:
                if _point_in_polygon(cx, cy, pts):
                    return item
        return None

    def _update_hover(self, cx: int, cy: int) -> None:
        new_hover = self._ann_item_at(cx, cy)
        if new_hover == self._hovered_item:
            return
        # Restore old hover (unless it's the selected item)
        if self._hovered_item and self._hovered_item != self._selected_item:
            norm_outline, norm_width = self.item_style.get(self._hovered_item, ("", 1))
            self.canvas.itemconfig(self._hovered_item, outline=norm_outline, width=norm_width)
        # Apply new hover
        if new_hover and new_hover != self._selected_item:
            _, norm_width = self.item_style.get(new_hover, ("", 1))
            self.canvas.itemconfig(new_hover, width=norm_width + 2)
        self._hovered_item = new_hover
        self.canvas.config(cursor="hand2" if new_hover else "crosshair")

    def _try_select(self, cx: int, cy: int) -> None:
        item = self._ann_item_at(cx, cy)
        if item == self._selected_item:
            return
        # Restore previously selected item
        if self._selected_item:
            norm_outline, norm_width = self.item_style.get(self._selected_item, ("", 1))
            self.canvas.itemconfig(self._selected_item, outline=norm_outline, width=norm_width)
        self._selected_item = item
        if item:
            self.canvas.itemconfig(item, outline="#FFFFFF", width=3)
        self._update_sidebar(self.item_data.get(item) if item else None)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _update_sidebar(self, ann: dict | None) -> None:
        self._sidebar.config(state=tk.NORMAL)
        self._sidebar.delete("1.0", tk.END)
        if ann:
            self._sidebar.insert(tk.END, f"ID\n{ann.get('id', '—')}\n\n")
            self._sidebar.insert(tk.END, f"Type\n{ann.get('type', '—')}\n\n")
            if ann.get("parent_id"):
                self._sidebar.insert(tk.END, f"Parent\n{ann['parent_id']}\n\n")
            if ann.get("text"):
                self._sidebar.insert(tk.END, f"Text\n{ann['text']}\n\n")
            extra = {k: v for k, v in ann.get("attrs", {}).items()}
            if extra:
                self._sidebar.insert(tk.END, "Attributes\n")
                for k, v in extra.items():
                    self._sidebar.insert(tk.END, f"  {k}: {v}\n")
        self._sidebar.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _classify_arg(path: str) -> str:
    """Return 'image', 'xml', or 'unknown' based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in ANNOTATION_EXTENSIONS:
        return "xml"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAGE XML Viewer — overlay PAGE XML annotations on a manuscript image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python page_viewer.py                          # use file dialogs\n"
            "  python page_viewer.py annotation.xml           # load XML only\n"
            "  python page_viewer.py image.jpg annotation.xml # pre-load both"
        ),
    )
    parser.add_argument(
        "files", nargs="*", metavar="FILE",
        help="Optional image and/or PAGE XML file paths to pre-load (order does not matter)",
    )
    args = parser.parse_args()

    preload_image = preload_xml = None
    for path in args.files:
        kind = _classify_arg(path)
        if kind == "image":
            preload_image = path
        elif kind == "xml":
            preload_xml = path
        else:
            parser.error(f"Cannot determine file type for: {path}")

    app = PageViewerApp(preload_xml=preload_xml, preload_image=preload_image)
    app.mainloop()


if __name__ == "__main__":
    main()
