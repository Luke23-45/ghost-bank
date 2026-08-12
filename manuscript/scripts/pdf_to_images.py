#!/usr/bin/env python3
"""Convert a PDF into per-page PNG images with zero-padded page numbers.

Defaults to the project's ``build\\main.pdf`` (project root is the parent of
this script's directory). A different PDF can be supplied via ``--pdf'', and a
different output directory via ``--out-dir`` (default: ``<pdf_dir>/images``).

Exit code is 0 on full success, 1 on fatal errors, 2 if the PDF exists but one
or more pages failed to render.
"""

import argparse
import os
import sys


def resolve_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def resolve_default_pdf():
    return os.path.join(resolve_project_root(), "build", "main.pdf")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="pdf_to_images",
        description="Render a PDF to one PNG image per page.",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to the PDF file (default: <repo>/build/main.pdf).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for the rendered images (default: <pdf_dir>/images).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render resolution in dots per inch (default: 200).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render pages even if the image file already exists.",
    )
    args = parser.parse_args(argv)
    if args.dpi < 20:
        parser.error("--dpi must be at least 20.")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    pdf_path = args.pdf if args.pdf else resolve_default_pdf()
    pdf_path = os.path.abspath(pdf_path)

    if not os.path.isfile(pdf_path):
        print("FATAL: PDF not found: {0}".format(pdf_path), file=sys.stderr)
        return 1

    if os.path.splitext(pdf_path)[1].lower() != ".pdf":
        print("FATAL: Not a .pdf file: {0}".format(pdf_path), file=sys.stderr)
        return 1

    out_dir = args.out_dir if args.out_dir else os.path.join(
        os.path.dirname(pdf_path), "images"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import fitz  # PyMuPDF (lazily imported; error reported cleanly below)
    except ImportError:
        print(
            "FATAL: PyMuPDF is required. Install it with: pip install pymupdf",
            file=sys.stderr,
        )
        return 1

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        print(
            "FATAL: Could not open PDF '{0}': {1}".format(pdf_path, exc),
            file=sys.stderr,
        )
        return 1

    page_count = document.page_count

    try:
        with document:
            errors = 0
            if page_count == 0:
                print("WARNING: PDF contains no pages: {0}".format(pdf_path))
                print("Output directory: {0}".format(out_dir))
                return 0

            width = len(str(page_count))
            for index in range(page_count):
                filename = "page_{0:0{1}d}.png".format(index + 1, width)
                file_path = os.path.join(out_dir, filename)

                if os.path.isfile(file_path):
                    if args.overwrite:
                        try:
                            os.remove(file_path)
                        except OSError as exc:
                            print(
                                "ERROR: Could not overwrite '{0}': {1}".format(
                                    file_path, exc
                                ),
                                file=sys.stderr,
                            )
                            errors += 1
                            continue
                    else:
                        continue

                try:
                    page = document.load_page(index)
                    matrix = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    pixmap.save(file_path)
                except Exception as exc:
                    print(
                        "ERROR: Failed to render page {0} to '{1}': {2}".format(
                            index + 1, file_path, exc
                        ),
                        file=sys.stderr,
                    )
                    errors += 1
                    continue

                print("Rendered {0}".format(filename), flush=True)

    except Exception as exc:
        print("FATAL: Unexpected failure: {0}".format(exc), file=sys.stderr)
        return 1

    rendered = page_count - errors
    print(
        "Done: {0} of {1} pages rendered to '{2}'.".format(
            rendered, page_count, out_dir
        )
    )
    if errors:
        print(
            "WARNING: {0} page(s) failed. See errors above.".format(errors),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())