"""Interactive annotation dashboard for sequential segmentation.

Three stacked panels: the image (draw boxes here), the predicted class map, and
the predictive entropy map. Draw a few boxes per class, hit "Train & Predict" to
refit and refresh the maps, repeat until satisfied, then "Save & Next Image".

Keyboard controls, with the cursor over the top panel:
    click & drag   draw a box for the active class
    d              delete the box under the cursor
    c              cycle the active class
"""

import json
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Rectangle

from ..boxes import combine_datasets, get_training_data_from_crops
from ..models import predict_with_uncertainty, train_and_optimize
from ..viz import plot_entropy_ax, plot_prediction_overlay_ax


class ActiveSegmentationDashboard:
    def __init__(self, loader, save_dir, mode="patch", track_progress=False):
        self.loader = loader
        self.save_dir = Path(save_dir)
        self.mode = mode
        self.track_progress = track_progress
        self.data = self.loader.next_image(mode=self.mode)

        self.boxes = []
        self.classes = {}
        self.class_names = []
        self.current_class_idx = 0
        self._dragging = False

        self._build_ui()
        self._load_current_image()

    def _build_ui(self):
        self.fig, (self.ax_img, self.ax_pred, self.ax_ent) = plt.subplots(3, 1, figsize=(9, 14))
        self.fig.canvas.header_visible = False
        self.fig.tight_layout(pad=3.0)

        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.instructions = widgets.HTML(
            value="<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px;'>"
                  "<b> Keyboard Controls (Hover over top image):</b> "
                  "&nbsp;&nbsp;|&nbsp;&nbsp; <b>Click & Drag</b>: Draw Box "
                  "&nbsp;&nbsp;|&nbsp;&nbsp; <b>'d'</b>: Delete Box (hover over it) "
                  "&nbsp;&nbsp;|&nbsp;&nbsp; <b>'c'</b>: Cycle Class"
                  "</div>"
        )

        self.text_class = widgets.Text(placeholder="Type class name...", description="New Class:")
        self.btn_add_class = widgets.Button(description="Add Class", button_style='info')
        self.btn_cycle = widgets.Button(description="Next Class")

        self.btn_train = widgets.Button(description="Train & Predict", button_style='warning')
        self.btn_next = widgets.Button(description="Save & Next Image", button_style='success')

        self.output_log = widgets.Output()

        self.btn_add_class.on_click(self._on_add_class)
        self.btn_cycle.on_click(self._on_cycle_class)
        self.btn_train.on_click(self._on_train_predict)
        self.btn_next.on_click(self._on_save_next)

        controls = widgets.HBox([self.text_class, self.btn_add_class, self.btn_cycle])
        actions = widgets.HBox([self.btn_train, self.btn_next])

        display(widgets.VBox([self.instructions, controls, actions, self.output_log]))

    def _load_current_image(self):
        self.round_counter = 0
        self.boxes = []
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / f"box_{self.data['img_id']}.json"

        self.ax_img.clear()
        self.ax_img.imshow(self.data['img_array'], cmap='gray', aspect='equal')
        self._refresh_title()
        self.ax_img.axis('off')

        if self.data.get('y_pred') is not None:
            plot_prediction_overlay_ax(self.ax_pred, self.data['img_array'], self.data['y_pred'],
                                       class_names=self.class_names)
            plot_entropy_ax(self.ax_ent, self.data['entropy'], self.data['img_array'].shape)
            with self.output_log:
                print(f"Loaded Zero-Shot Prediction for {self.data['img_id']}")
            if self.track_progress:
                self._save_progress_state()
        else:
            self.ax_pred.clear(); self.ax_pred.set_title("Prediction (Awaiting Training)"); self.ax_pred.axis('off')
            self.ax_ent.clear(); self.ax_ent.set_title("Entropy (Awaiting Training)"); self.ax_ent.axis('off')

        self.fig.canvas.draw_idle()

    def _on_add_class(self, b):
        name = self.text_class.value.strip()
        if name and name not in self.classes:
            class_idx = len(self.class_names)
            cmap_colors = plt.get_cmap('tab20').colors
            color = cmap_colors[class_idx % 20]

            self.classes[name] = color
            self.class_names.append(name)
            self.current_class_idx = class_idx
            self.text_class.value = ""
            self._refresh_title()

    def _on_cycle_class(self, b):
        if self.class_names:
            self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
            self._refresh_title()

    def _on_train_predict(self, b):
        with self.output_log:
            self.round_counter += 1
            self.output_log.clear_output()
            print("Training classifier...")
            clean_boxes = [{"x0": b["x0"], "y0": b["y0"], "x1": b["x1"], "y1": b["y1"], "cls": b["cls"]}
                           for b in self.boxes]

            with open(self.save_path, "w") as f:
                json.dump(clean_boxes, f)

            self.X_train_c, self.y_train_c, self.X_train_c_original = get_training_data_from_crops(
                self.data,
                self.save_path,
                class_names=self.class_names
            )

            if self.data.get('X_train') is None:
                self.X_train_combined, self.y_train_combined = self.X_train_c, self.y_train_c
            else:
                self.X_train_combined, self.y_train_combined = combine_datasets(
                    self.X_train_c, self.y_train_c, self.data['X_train'], self.data['y_train'])

            best_model = train_and_optimize(self.X_train_combined, self.y_train_combined)
            if self.mode == 'pixel':
                inference_features = self.data['features_pixel_normalized']
            else:
                inference_features = self.data['features_normalized']

            self.y_pred, self.probs, self.entropy = predict_with_uncertainty(best_model, inference_features)

            plot_prediction_overlay_ax(self.ax_pred, self.data['img_array'], self.y_pred,
                                       class_names=self.class_names)
            plot_entropy_ax(self.ax_ent, self.entropy, self.data['img_array'].shape)
            self.fig.canvas.draw_idle()
            print(f"Prediction updated in {self.mode.upper()} mode!")

            if self.track_progress:
                self._save_progress_state()

    def _on_save_next(self, b):
        with self.output_log:
            print("Saving state and loading next image...")
            # Fall back to the loaded zero-shot prediction when the user accepted
            # it without retraining (self.y_pred/probs/entropy are only set by
            # _on_train_predict). Note the loader stores probabilities under the
            # key 'probabilities'.
            self.loader.save(
                y_pred=getattr(self, 'y_pred', self.data.get('y_pred')),
                probs=getattr(self, 'probs', self.data.get('probabilities')),
                entropy=getattr(self, 'entropy', self.data.get('entropy')),
                X_train_c_original=getattr(self, 'X_train_c_original', None),
                y_train_c=getattr(self, 'y_train_c', None)
            )

            for attr in ['y_pred', 'probs', 'entropy', 'X_train_c_original', 'y_train_c']:
                if hasattr(self, attr):
                    delattr(self, attr)

            next_data = self.loader.next_image(mode=self.mode)
            if next_data is None:
                print("Experiment Complete. No more images.")
                return

            self.data = next_data
            self._load_current_image()

    def _current_class(self):
        if not self.class_names: return None, None
        name = self.class_names[self.current_class_idx]
        return name, self.classes[name]

    def _refresh_title(self):
        name, color = self._current_class()
        title = f"ACTIVE: {name.upper()}" if name else "ADD A CLASS TO START"
        self.ax_img.set_title(title, color=color or 'red', fontweight='bold')
        self.fig.canvas.draw_idle()

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax_img: return
        name, color = self._current_class()
        if not name: return
        self._dragging = True
        self._x0, self._y0 = event.xdata, event.ydata
        self._preview_patch = Rectangle((self._x0, self._y0), 0, 0, fill=False, ec=color, ls='--')
        self.ax_img.add_patch(self._preview_patch)

    def _on_mouse_move(self, event):
        if not self._dragging or event.inaxes != self.ax_img: return
        x1, y1 = event.xdata, event.ydata
        self._preview_patch.set_xy((min(self._x0, x1), min(self._y0, y1)))
        self._preview_patch.set_width(abs(x1 - self._x0))
        self._preview_patch.set_height(abs(y1 - self._y0))
        self.fig.canvas.draw_idle()

    def _on_mouse_release(self, event):
        if not self._dragging: return
        self._dragging = False
        if event.inaxes == self.ax_img:
            x, y = min(self._x0, event.xdata), min(self._y0, event.ydata)
            w, h = abs(event.xdata - self._x0), abs(event.ydata - self._y0)
            if w > 1 and h > 1:
                name, color = self._current_class()
                rect = Rectangle((x, y), w, h, fill=False, ec=color, lw=2)
                self.ax_img.add_patch(rect)
                self.boxes.append({"x0": int(x), "y0": int(y), "x1": int(x + w), "y1": int(y + h),
                                   "cls": name, "patch": rect})
        if self._preview_patch: self._preview_patch.remove(); self._preview_patch = None
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'd':
            for i, b in enumerate(self.boxes):
                if (b["x0"] <= event.xdata <= b["x1"]) and (b["y0"] <= event.ydata <= b["y1"]):
                    b["patch"].remove()
                    self.boxes.pop(i)
                    break
            self.fig.canvas.draw_idle()

        elif event.key == 'c':
            self._on_cycle_class(None)

    def _save_progress_state(self):
        """Save this round's boxes, predictions and entropy for later analysis."""
        if not self.track_progress:
            return

        img_id = self.data['img_id']

        prog_dir = self.save_dir / "progress" / img_id
        prog_dir.mkdir(parents=True, exist_ok=True)

        boxes_path = prog_dir / f"round_{self.round_counter}_boxes.json"
        clean_boxes = [{"x0": b["x0"], "y0": b["y0"], "x1": b["x1"], "y1": b["y1"], "cls": b["cls"]}
                       for b in self.boxes]
        with open(boxes_path, "w") as f:
            json.dump(clean_boxes, f)

        current_y_pred = getattr(self, 'y_pred', self.data.get('y_pred'))
        current_entropy = getattr(self, 'entropy', self.data.get('entropy'))

        if current_y_pred is not None and current_entropy is not None:
            preds_path = prog_dir / f"round_{self.round_counter}_preds.npz"
            np.savez_compressed(preds_path, y_pred=current_y_pred, entropy=current_entropy)

        with self.output_log:
            print(f"[Progress Tracker] Saved Round {self.round_counter} for {img_id}")
