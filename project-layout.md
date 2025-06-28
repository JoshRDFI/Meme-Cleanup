
# project-layout.md

## project-overview

This project is a high-performance, GPU-accelerated image deduplication and organization tool with a PyQt graphical interface. It scans multiple large directories of images, visually matches and deduplicates them using CLIP embeddings and image quality metrics (BRISQUE/NIQE), and consolidates them into a single, user-specified directory tree. The tool is modular, supports progress saving, and is designed for extensibility (e.g., face matching, image enhancement).

## user-flow

1. **Startup:**  
   User launches the application and is greeted with a landing screen to select source directories and an output location.

2. **Configuration:**  
   User sets deduplication preferences (quality metric, resource usage, etc.) and starts the scan.

3. **Processing:**  
   The app processes images in batches, displaying real-time progress and allowing the user to pause or save progress.

4. **Review:**  
   After processing, the user is presented with a summary of duplicates per subdirectory. For images found in multiple subdirectories, the user can choose which to keep and where.

5. **Finalize:**  
   User confirms choices, and the app consolidates images into the output directory, merging subdirectories as needed.

6. **Completion:**  
   User can view a summary report onscreen, export results, or resume later if not finished.

## tech-stack

- **Python 3.12:** Core language for all logic and scripting.
- **PyQt:** GUI framework for cross-platform, native-feeling desktop interface.
- **CUDA (Nvidia):** GPU acceleration for embedding and quality metric computation.
- **PyTorch (Nightly, with CUDA):** For latest GPU support and running CLIP and quality models.
- **CLIP (OpenAI):** Visual embedding model for image similarity.
- **BRISQUE/NIQE:** No-reference image quality metrics for selecting the cleanest image.
- **SQLite:** Local database for storing image metadata, embeddings, and progress.
- **Pillow, OpenCV:** Image loading, processing, and metadata handling.

## ui-rules

- Use PyQt’s Model/View architecture for scalable, responsive lists and tables.
- Tabs for: Progress, Review Duplicates, Settings, and Logs.
- All long-running operations must be threaded or run in background processes to keep the UI responsive.
- Duplicates review screen must allow side-by-side image comparison, metadata display, and easy selection.
- All destructive actions (deletion, overwrite) require confirmation.
- Progress and status indicators must be visible at all times.
- Save/resume progress is always available from the main menu.

## theme-rules

- **Colors:**  
  - Primary: #2D2D2D (dark gray), #4A90E2 (blue accent)  
  - Secondary: #F5F5F5 (light background), #E94E77 (error/red accent)
- **Typography:**  
  - Use a clean, sans-serif font (e.g., Segoe UI, Roboto, or system default).
  - Headings: Bold, 1.2x size of body text.
  - Body: Regular, 12-14pt.
- **Spacing:**  
  - Minimum 8px padding between elements.
  - Consistent margins for all containers.
- **Component Style:**  
  - Buttons: Rounded corners, clear hover/focus states.
  - Lists/Tables: Alternating row colors for readability.
  - Dialogs: Modal, with clear action buttons.

## project-rules

- **Folder Structure:**  
  - `src/` — All source code  
    - `gui/` — PyQt UI components  
    - `core/` — Deduplication, embedding, quality, and file operations  
    - `db/` — Database models and helpers  
    - `utils/` — Utility functions (image loading, metadata, threading, etc.)  
    - `modules/` — Optional/extendable features (face matching, enhancement, etc.)
  - `tests/` — Unit and integration tests
  - `_docs/` — Documentation  
    - `phases/` — Project phases and feature/task breakdowns
  - `requirements.txt` — Python dependencies
  - `README.md` — Project introduction and quickstart
  - `project-layout.md` — This document

- **File Naming:**  
  - Use `snake_case` for Python files and folders.
  - Use `CamelCase` for class names, `snake_case` for functions and variables.
  - All modules must have a docstring at the top describing their purpose.

- **Code Style:**  
  - Follow PEP8 for Python code.
  - All public functions and classes must have type hints and docstrings.
  - Use logging for all non-UI errors and warnings.

## _docs/phases/

Each phase document should include:  
- **Phase Overview:** What this phase accomplishes  
- **Tasks:** List of tasks and features  
- **Dependencies:** What must be completed first  
- **Acceptance Criteria:** What defines “done” for this phase

**Example phase documents:**

- `phase_01_initial_scan.md`  
  - Scanning directories, loading images, storing metadata/embeddings in SQLite

- `phase_02_embedding_and_quality.md`  
  - Running CLIP, BRISQUE/NIQE, storing results

- `phase_03_duplicate_detection.md`  
  - Comparing embeddings, flagging duplicates, handling edge cases

- `phase_04_gui_review.md`  
  - Building the review interface, user selection, progress saving

- `phase_05_file_operations.md`  
  - Copying/merging files, metadata transfer, error handling

- `phase_06_modular_extensions.md`  
  - Adding new modules (face matching, enhancement, etc.)
