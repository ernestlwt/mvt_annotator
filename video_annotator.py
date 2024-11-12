import base64
import glob
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from natsort import natsorted
from pycocotools import mask as coco_mask
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm
from ultralytics import YOLO


# Define the JsonSerializable abstract base class
class JsonSerializable(ABC):
    @abstractmethod
    def to_json(self):
        """Serializes the object to a JSON-compatible dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data):
        """Deserializes the object from a JSON-compatible dictionary."""
        pass


@dataclass
class Mask(JsonSerializable):
    """
    Represents a mask with an encoded string and shape.
    """

    encoded: str  # Encoded string as per pycocotools
    shape: List[int]  # [height, width]

    def to_json(self):
        json_encoded = self.encoded.copy()
        json_encoded["counts"] = base64.b64encode(self.encoded["counts"]).decode(
            "utf-8"
        )

        return {"encoded": json_encoded, "shape": self.shape}

    @classmethod
    def from_json(cls, data):
        json_encoded = data["encoded"].copy()
        json_encoded["counts"] = base64.b64decode(json_encoded["counts"])

        return cls(encoded=json_encoded, shape=data["shape"])


@dataclass
class ObjectDetection(JsonSerializable):
    """
    Stores the bounding box and class of an object.
    """

    detection_id: int
    xyxyn: List[
        float
    ]  # Normalized bounding box coordinates [x_min, y_min, x_max, y_max]
    object_class: str  # Class of the object, e.g., 'human' or 'vehicle'

    def to_json(self):
        return {
            "detection_id": self.detection_id,
            "xyxyn": self.xyxyn,
            "object_class": self.object_class,
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            detection_id=data["detection_id"],
            xyxyn=data["xyxyn"],
            object_class=data["object_class"],
        )


@dataclass
class ObjectTracking(JsonSerializable):
    """
    Stores the object tracking information.
    """

    tracking_id: int
    start_frame: int
    duration_frames: int
    masks: Dict[int, Mask] = field(default_factory=dict)  # Frame ID to Mask
    original_detection_id: Dict[int, Optional[int]] = field(
        default_factory=dict
    )  # Frame ID to detection ID

    def to_json(self):
        return {
            "tracking_id": self.tracking_id,
            "start_frame": self.start_frame,
            "duration_frames": self.duration_frames,
            "masks": {
                str(frame_id): mask.to_json() for frame_id, mask in self.masks.items()
            },
            "original_detection_id": {
                str(frame_id): det_id
                for frame_id, det_id in self.original_detection_id.items()
            },
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            tracking_id=data["tracking_id"],
            start_frame=data["start_frame"],
            duration_frames=data["duration_frames"],
            masks={
                int(frame_id): Mask.from_json(mask_data)
                for frame_id, mask_data in data.get("masks", {}).items()
            },
            original_detection_id={
                int(frame_id): det_id
                for frame_id, det_id in data.get("original_detection_id", {}).items()
            },
        )


@dataclass
class SingleVideoAnnotatorState(JsonSerializable):
    """
    State of the video annotator focusing on a single video.
    """

    frame_object_detections: Dict[int, List[ObjectDetection]] = field(
        default_factory=dict
    )  # Frame ID to list of detections
    object_trackings: Dict[int, ObjectTracking] = field(
        default_factory=dict
    )  # Tracking ID to ObjectTracking

    num_assigned_detections: int = 0  # Number of assigned detections
    num_assigned_trackings: int = 0  # Number of assigned trackings

    def to_json(self):
        return {
            "frame_object_detections": {
                str(frame_id): [detection.to_json() for detection in detections]
                for frame_id, detections in self.frame_object_detections.items()
            },
            "object_trackings": {
                str(tracking_id): tracking.to_json()
                for tracking_id, tracking in self.object_trackings.items()
            },
            "num_assigned_detections": self.num_assigned_detections,
            "num_assigned_trackings": self.num_assigned_trackings,
        }

    @classmethod
    def from_json(cls, data):
        frame_object_detections = {
            int(frame_id): [
                ObjectDetection.from_json(det_data) for det_data in detections
            ]
            for frame_id, detections in data.get("frame_object_detections", {}).items()
        }
        object_trackings = {
            int(tracking_id): ObjectTracking.from_json(tracking_data)
            for tracking_id, tracking_data in data.get("object_trackings", {}).items()
        }
        num_assigned_detections = data.get("num_assigned_detections", 0)
        num_assigned_trackings = data.get("num_assigned_trackings", 0)
        return cls(
            frame_object_detections=frame_object_detections,
            object_trackings=object_trackings,
            num_assigned_detections=num_assigned_detections,
            num_assigned_trackings=num_assigned_trackings,
        )


# Define the VideoSource abstract base class
class VideoSource(ABC):
    @abstractmethod
    def get_frame(self, frame_index: int):
        """Returns the frame at the given index."""
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Returns the total number of frames in the video."""
        pass

    @abstractmethod
    def release(self):
        """Releases any resources held by the video source."""
        pass


class ImageFolderSource(VideoSource):
    def __init__(self, folder_path: str, sorting_rule: Optional[callable] = None):
        self.folder_path = folder_path
        # Supported image extensions
        image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        if not self.image_paths:
            raise IOError(f"No images found in folder: {folder_path}")
        # Apply sorting rule
        # self.image_paths.sort(key=sorting_rule if sorting_rule else lambda x: x)

        self.image_paths = sorted(self.image_paths, key=sorting_rule)

        self.frame_count = len(self.image_paths)

    def get_frame(self, frame_index: int):
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError("Frame index out of range")
        image_path = self.image_paths[frame_index]
        frame = cv2.imread(image_path)
        if frame is None:
            raise IOError(f"Failed to read image at path: {image_path}")
        return frame

    def get_frame_count(self) -> int:
        return self.frame_count

    def release(self):
        pass  # No resource to release for image folders

    def get_image_paths(self, start_index: int, end_index: int) -> List[str]:
        """
        Returns a list of image paths between start_index (inclusive) and end_index (exclusive).
        """
        if start_index < 0 or end_index > self.frame_count or start_index >= end_index:
            raise ValueError("Invalid index range")
        return self.image_paths[start_index:end_index]


class SingleVideoAnnotatorModel:
    """
    Model of the video annotator application responsible for storing:
        - Detections
        - Single-video object tracking (without re-identification)
    The model notifies observers when its state changes and exposes an interface for the controller.
    """

    def __init__(
        self,
        video_id: str,
        state_save_folder: str,
        video_source_path: str,
        sorting_rule: Optional[callable] = None,
        object_classes: Set[str] = {"person"},
    ):
        """
        Initializes the annotator model for a single video.

        Args:
            video_id (str): Identifier for the video being annotated.
            state_save_folder (str): Folder to save the state of the annotator.
            video_source_path (str): Path to the video file or image folder.
            sorting_rule (callable, optional): Sorting function for image filenames if video_source_path is an image folder.
        """
        self.video_id = video_id
        self.state_save_folder = state_save_folder
        self.observers: List = []

        self.object_classes = object_classes

        self.yolo_model = YOLO("yolov10x.pt").cuda()  # Load an official Detect model

        # Determine the video source based on the file extension
        if os.path.isfile(video_source_path) and video_source_path.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):
            raise NotImplementedError("Video file source is not supported yet")
        elif os.path.isdir(video_source_path):
            self.video_source = ImageFolderSource(
                video_source_path, sorting_rule=sorting_rule
            )
        else:
            raise ValueError(f"Invalid video source path: {video_source_path}")

        self.state = SingleVideoAnnotatorState()

        self.image_shape = self.get_frame(0).shape[:2]  # H, W, 3

        # Ensure the save folder exists
        os.makedirs(self.state_save_folder, exist_ok=True)

    def add_observer(self, observer):
        """
        Adds an observer that will be notified when the state changes.
        """
        self.observers.append(observer)

    def remove_observer(self, observer):
        """
        Removes an observer.
        """
        self.observers.remove(observer)

    def notify_observers(self, **kwargs):
        """
        Notifies all observers about a state change.
        """
        for observer in self.observers:
            observer.update(self.state, **kwargs)

    # state modification methods
    def add_detection(self, frame_id: int, detection: ObjectDetection):
        """
        Adds a detection to a specific frame.
        """
        detections = self.state.frame_object_detections.setdefault(frame_id, [])
        detections.append(detection)
        # self.notify_observers(
        #     frame_id=frame_id,
        #     changed="detections"
        # )

    def clear_detections(self, frame_id: int):
        """
        Clears all detections from a specific frame.
        """
        self.state.frame_object_detections.pop(frame_id, None)
        # self.notify_observers(
        #     frame_id=frame_id,
        #     changed="detections"
        # )

    def append_mask_to_tracking(self, tracking_id: int, frame_id: int, mask: Mask):
        """
        Appends a mask to an existing tracking.
        """
        if tracking_id not in self.state.object_trackings:
            self.add_tracking(
                ObjectTracking(
                    tracking_id=tracking_id, start_frame=frame_id, duration_frames=0
                )
            )

        tracking = self.state.object_trackings[tracking_id]
        tracking.masks[frame_id] = mask
        tracking.duration_frames = max(
            tracking.duration_frames, frame_id - tracking.start_frame
        )

        # self.notify_observers(
        #     frame_id=frame_id,
        #     tracking_id=tracking_id,
        #     changed="tracking"
        # )

    def add_tracking(self, tracking: ObjectTracking):
        """
        Adds a tracking object.
        """
        self.state.object_trackings[tracking.tracking_id] = tracking
        self.notify_observers(tracking_id=tracking.tracking_id, changed="tracking")

    # the actual tracking logics.
    def initialize_YOLO_detection(self):
        """
        Run YOLO detection on the entire video and save the decections.

        Although YOLO have a tracking method, we will not use it here.
        We will use SAM to provide tracking.
        """

        # Run YOLO detection on the entire video
        for frame_id in tqdm(range(self.get_frame_count())):
            frame = self.get_frame(frame_id)
            detection_result = self.yolo_model.track(
                frame, imgsz=2560, persist=True, show=False, verbose=False
            )[0]

            detected_boxes = detection_result.boxes
            det_class_mapping = detection_result.names

            self.clear_detections(frame_id)

            for detection in detected_boxes:
                new_detection_id = self.assign_new_detection_id()
                detected_class = det_class_mapping[int(detection.cls[0].item())]

                # check if the detected class is in the object classes
                if detected_class not in self.object_classes:
                    continue

                self.add_detection(
                    frame_id,
                    ObjectDetection(
                        detection_id=new_detection_id,
                        xyxyn=detection.xyxyn[0].tolist(),
                        object_class=detected_class,
                    ),
                )

            self.notify_observers(frame_id=frame_id, changed="detections")

    def initialize_SAM_tracking(self, sam_batchsize: int = 1):
        # sam2_checkpoint = "segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
        sam2_checkpoint = "segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
        # model_cfg = "sam2_hiera_b+.yaml"
        model_cfg = "sam2_hiera_t.yaml"
        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device="cuda:0"
        )

        current_frame = 0

        sam_range_start = -1
        sam_range_end = -1

        with torch.autocast("cuda", torch.bfloat16):
            while current_frame < self.get_frame_count() - 1:

                print("Current frame:", current_frame)

                # prepare SAM prompts at current frame
                sam_prompts = self.prepare_sam_prompts(current_frame)

                if len(sam_prompts) == 0:
                    current_frame += 1
                    continue  # if no prompts is possible, skip the frame

                # initialize predictor state if we switched to a new batch
                if current_frame >= sam_range_end:
                    sam_range_start = current_frame
                    sam_range_end = current_frame + sam_batchsize

                    # limit the end frame to the total frame count
                    sam_range_end = min(sam_range_end, self.get_frame_count() - 1)

                    image_paths = self.video_source.get_image_paths(
                        sam_range_start, sam_range_end + 1
                    )  # predict 1 more frame because we want 1 overlap to propagate the masks
                    predictor_state = predictor.init_state(frame_paths=image_paths)

                predictor.reset_state(predictor_state)

                for tracking_id, mask_bbox in sam_prompts:
                    # add the prompt to the predictor
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=predictor_state,
                        frame_idx=current_frame - sam_range_start,
                        obj_id=tracking_id,
                        box=mask_bbox,
                    )

                # propagate the masks along the video. upon receiving the masks,
                # test can current detections be explained by the masks. If there
                # are unexplained detections, stop the propagation and add them to
                # the SAM prompts.
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(predictor_state):
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                        mask_arr = np.asfortranarray(mask)

                        if mask_arr.any():
                            mask_obj = Mask(
                                encoded=coco_mask.encode(np.asfortranarray(mask_arr)),
                                shape=mask_arr.shape,
                            )

                            # add the mask to the tracking
                            self.append_mask_to_tracking(
                                out_obj_id, out_frame_idx + sam_range_start, mask_obj
                            )

                    self.notify_observers(
                        frame_id=out_frame_idx + sam_range_start,
                        tracking_id=tracking_id,
                        changed="tracking",
                    )

                    # explain the detections with the masks
                    unexplained_detections = self.get_unexplained_detections_at_frame(
                        out_frame_idx + sam_range_start
                    )

                    if len(unexplained_detections) > 0:
                        break  # stop the propagation

                # update the current frame and repeat the process
                current_frame = out_frame_idx + sam_range_start
                self.save_state()

    def prepare_sam_prompts(self, frame_id: int):
        """
        Prepares SAM prompts for a frame.

        Args:
            frame_id (int): Frame ID for which to prepare the prompts.

        Returns:
            List[str]: List of prompts for SAM.
        """
        # For all object tracking, get the masks for the frame. These tracking ids # will be kept in the SAM prompts.
        annotated_masks = self.get_annotated_masks_at_frame(frame_id)
        unexplained_detections = self.get_unexplained_detections_at_frame(frame_id)

        # prepare the SAM prompts based on existing masks and unexplained detections. The prompts will be in the form of bounding boxes.
        sam_prompts = []

        for tracking_id, mask in annotated_masks.items():
            mask_bbox = coco_mask.toBbox(mask.encoded)

            # notice! this bbox is in x, y, w, h format
            mask_bbox_xyxy = np.array(
                [
                    mask_bbox[0],
                    mask_bbox[1],
                    mask_bbox[0] + mask_bbox[2],
                    mask_bbox[1] + mask_bbox[3],
                ]
            )

            sam_prompts.append((tracking_id, mask_bbox_xyxy))

        for detection in unexplained_detections:
            detection_bbox = detection.xyxyn * np.array(
                [
                    self.image_shape[1],
                    self.image_shape[0],
                    self.image_shape[1],
                    self.image_shape[0],
                ]
            )
            sam_prompts.append((self.assign_new_tracking_id(), detection_bbox))

        return sam_prompts

    def get_annotated_masks_at_frame(self, frame_id: int):
        annotated_masks = {}  # tracking_id to mask
        for tracking_id, tracking in self.state.object_trackings.items():
            if frame_id in tracking.masks:
                mask = tracking.masks[frame_id]
                # Add the mask to the SAM prompts
                annotated_masks[tracking_id] = mask

        return annotated_masks

    def get_unexplained_detections_at_frame(self, frame_id: int):

        annotated_masks = self.get_annotated_masks_at_frame(frame_id)

        # Get the detections for the frame, try to explain them with the masks. If not possible, add the detection to the SAM prompts with a new tracking id.
        all_detections = self.state.frame_object_detections.get(frame_id, [])

        # explain the detections with the masks
        unexplained_detections = []
        for detection in all_detections:
            explained = False
            for tracking_id, mask in annotated_masks.items():
                if self.explain_detection_with_mask(detection, mask):
                    explained = True
                    break
            if not explained:
                unexplained_detections.append(detection)

        return unexplained_detections

    def explain_detection_with_mask(self, detection: ObjectDetection, mask: Mask):
        """
        Explains a detection with a mask.

        Args:
            detection (ObjectDetection): Detection to explain.
            mask (Mask): Mask to explain the detection.

        Returns:
            bool: True if the detection was explained, False otherwise.
        """

        # a detection is explained by a mask if the bounding of the mask
        # and the detection have error less then 0.1 normalized error.

        # get the bounding box of the mask
        decoded_mask = coco_mask.decode(mask.encoded)
        mask_bbox = coco_mask.toBbox(mask.encoded)  # notice! this is x, y, w, h

        detection_xyxy = np.array(
            [
                int(detection.xyxyn[0] * self.image_shape[1]),
                int(detection.xyxyn[1] * self.image_shape[0]),
                int(detection.xyxyn[2] * self.image_shape[1]),
                int(detection.xyxyn[3] * self.image_shape[0]),
            ]
        )

        mask_in_bbox = np.sum(
            decoded_mask[
                detection_xyxy[1] : detection_xyxy[3],
                detection_xyxy[0] : detection_xyxy[2],
            ]
        )
        mask_pixels = np.sum(decoded_mask)

        mask_in_bbox = mask_in_bbox / (mask_pixels + 1e-6)

        return mask_in_bbox > 0.6

    def assign_new_detection_id(self):
        """
        Assigns a new detection ID for a new detection.
        """
        assigned_id = self.state.num_assigned_detections

        self.state.num_assigned_detections += 1
        self.notify_observers()

        return assigned_id

    def assign_new_tracking_id(self):
        """
        Assigns a new tracking ID for a new tracking.
        """
        assigned_id = self.state.num_assigned_trackings

        self.state.num_assigned_trackings += 1
        self.notify_observers()

        return assigned_id

    # state writing methods
    def save_state(self):
        """
        Saves the current state to a JSON file.
        """
        state_file = os.path.join(self.state_save_folder, f"{self.video_id}_state.json")
        with open(state_file, "w") as f:
            json.dump(self.state.to_json(), f, indent=4)

    def load_state(self):
        """
        Loads the state from a JSON file.
        """
        state_file = os.path.join(self.state_save_folder, f"{self.video_id}_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state_dict = json.load(f)
                self.state = SingleVideoAnnotatorState.from_json(state_dict)
                self.notify_observers()

    def get_frame(self, frame_index: int):
        """
        Retrieves a frame from the video source.

        Args:
            frame_index (int): Index of the frame to retrieve.

        Returns:
            The video frame as an image.
        """
        return self.video_source.get_frame(frame_index)

    def get_frame_count(self) -> int:
        """
        Returns the total number of frames in the video.

        Returns:
            int: Total number of frames.
        """
        return self.video_source.get_frame_count()

    def release_video(self):
        """
        Releases the video source resources.
        """
        self.video_source.release()

    def get_all_tracking(self):
        """
        Returns all the tracking objects.
        """
        return self.state.object_trackings

    def split_tracking(self, tracking_id: int, split_frame: Tuple[int, int]):
        """
        Splits a tracking object at a specific frame.

        Args:
            tracking_id (int): ID of the tracking object to split.
            split_frame (int): Frame ID at which to split the tracking.
        """
        tracking = self.state.object_trackings[tracking_id]
        if tracking.start_frame == split_frame[1]:
            return

        new_tracking_id = self.assign_new_tracking_id()
        new_tracking = ObjectTracking(
            tracking_id=new_tracking_id,
            start_frame=split_frame[1],
            duration_frames=tracking.duration_frames
            - (split_frame[1] - tracking.start_frame),
            masks={
                frame_id: mask
                for frame_id, mask in tracking.masks.items()
                if frame_id >= split_frame[1]
            },
            original_detection_id={
                frame_id: det_id
                for frame_id, det_id in tracking.original_detection_id.items()
                if frame_id >= split_frame[0]
            },
        )

        tracking.duration_frames = split_frame[0] + 1 - tracking.start_frame
        # delete the masks and original detection ids from the original tracking
        tracking.masks = {
            frame_id: mask
            for frame_id, mask in tracking.masks.items()
            if frame_id <= split_frame[0]
        }
        tracking.original_detection_id = {
            frame_id: det_id
            for frame_id, det_id in tracking.original_detection_id.items()
            if frame_id <= split_frame[0]
        }

        self.add_tracking(new_tracking)

        return new_tracking_id

    # Additional methods for the controller to interact with the model can be added here

    def merge_tracking(self, tracking_id_1: int, tracking_id_2: int):
        """
        Merges two tracking objects.

        Args:
            tracking_id_1 (int): ID of the first tracking object.
            tracking_id_2 (int): ID of the second tracking object.
        """
        tracking_1 = self.state.object_trackings[tracking_id_1]
        tracking_2 = self.state.object_trackings[tracking_id_2]

        # merge the masks and original detection ids
        tracking_1.masks.update(tracking_2.masks)
        tracking_1.original_detection_id.update(tracking_2.original_detection_id)

        # delete the second tracking object
        self.state.object_trackings.pop(tracking_id_2)

        self.notify_observers()


import time


class SingleVideoAnnotatorView:
    def __init__(
        self, model: SingleVideoAnnotatorModel, update_rate_limit=10, save=False
    ):
        self.model = model
        self.model.add_observer(self)

        self.update_rate_limit = update_rate_limit
        self.update_interval = 0.0
        self.time_at_last_update = None

        self.save = save

        self.viz_frame_count = 0

        self.visualized = set()

    def update(self, state, **kwargs):
        """
        Updates the view when the model state changes.
        """

        if (
            self.time_at_last_update is None
            or time.time() - self.time_at_last_update > self.update_interval
        ):
            self.time_at_last_update = time.time()
            self._update(state, **kwargs)

    def _update(self, state, **kwargs):
        """
        Updates the view when the model state changes.
        """

        # render the most recently changed frame

        if "frame_id" in kwargs:
            frame_id = kwargs["frame_id"]
            changed = kwargs["changed"]

            self.viz_frame_count += 1

            frame_viz = self.draw_frame(state, frame_id)

            # resize frame viz into smaller size
            frame_viz = cv2.resize(frame_viz, (1920, 1080))

            # save the frame visualization
            if self.save:
                cv2.imwrite(f"viz_output/frame_{self.viz_frame_count}.png", frame_viz)

    def draw_frame(self, state, frame_id):
        """
        Draws the frame with annotations.
        """

        frame = self.model.get_frame(frame_id)

        # add frame number to the frame
        cv2.putText(
            frame,
            f"Frame {frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # draw detections
        detections = state.frame_object_detections.get(frame_id, [])

        H, W, _ = frame.shape

        for detection in detections:
            x_min, y_min, x_max, y_max = detection.xyxyn

            x_min, y_min, x_max, y_max = (
                int(x_min * W),
                int(y_min * H),
                int(x_max * W),
                int(y_max * H),
            )

            det_color = (255, 255, 255)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), det_color, 2)

        # draw masks at this frame
        all_masks = [
            (tracking_id, tracking.masks[frame_id])
            for tracking_id, tracking in state.object_trackings.items()
            if frame_id in tracking.masks
        ]
        for track_id, mask in all_masks:
            # draw the mask on the frame
            decoded_mask = coco_mask.decode(mask.encoded)
            color = np.array(self.get_color_from_id(track_id))

            colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
            colored_mask[decoded_mask > 0] = color

            # draw the mask on the frame
            frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0, frame)

        frame = cv2.resize(frame, (1920, 1080))

        return frame

    def get_color_from_id(self, tracking_id):
        """
        Returns a unique color for a tracking ID.
        """
        # Use the tracking ID to generate a hue value between 0 and 1
        hue = (tracking_id * 37) % 180  # OpenCV uses hue range [0, 179]

        # Create an HSV image with full saturation and value for bright colors
        hsv_color = np.uint8([[[hue, 255, 255]]])  # Hue, Saturation, Value

        # Convert the HSV color to BGR using OpenCV
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

        # Convert to RGB format
        r, g, b = int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])

        return (r, g, b)

    def __del__(self):
        cv2.destroyAllWindows()


from itertools import combinations

import matplotlib.pyplot as plt


class FrameClickHandler:
    def __init__(self, fig, axs, num_clicks=2):
        self.fig = fig  # Store the figure object
        self.axs = axs  # Store the axes
        self.clicks = []  # Store the indices of the clicked axes
        self.cid_click = None  # To store the connection ID for the click event
        self.cid_key = None  # To store the connection ID for the key press event
        self.continue_execution = False  # Control flag
        self.cancel_selection = False  # Flag to indicate if selection is canceled
        self.should_delete = (
            False  # Flag to indicate if the selection should be deleted
        )

        self.num_clicks = num_clicks

    def on_click(self, event):
        # Check if the click was on one of the axes
        for i, ax in enumerate(self.axs.flat):
            if ax == event.inaxes:
                print(f"Axis {i} clicked")  # Log the axis index

                # Add to the list of clicks
                self.clicks.append(i)

                # Check if two clicks have been registered
                if len(self.clicks) == self.num_clicks:
                    self.continue_execution = True  # Signal to continue execution
                break

    def on_key_press(self, event):
        # Check if the key pressed is 'n'
        if event.key == "n":
            print("Selection canceled by user.")
            self.cancel_selection = True
            self.continue_execution = True  # Signal to exit the loop
        elif event.key == "d":
            print("Selection should be deleted.")
            self.should_delete = True
            self.continue_execution = True

    def wait_for_clicks(self):
        # Reset clicks and connect the click and key press events
        self.clicks = []
        self.continue_execution = False  # Reset the flag
        self.cancel_selection = False  # Reset cancel flag
        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Start the event loop to capture clicks or 'n' press
        while not self.continue_execution:
            plt.pause(0.1)  # Small pause to keep the GUI responsive

        # Disconnect the event handlers
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_key)

        # Return the collected indices if not canceled
        return None if self.cancel_selection else self.clicks


class SingleVideoAnnotatorController:
    def __init__(
        self, model: SingleVideoAnnotatorModel, view: SingleVideoAnnotatorView
    ):
        self.model = model
        self.view = view

    """
    Guided labeling mode will present a continuous object tracking with sampling
    along the frames. The user will be asked if the human along this annotation
    is the same human. If not, where should the tracking be split. Use a matplotlib
    interface for the user to interact with the video.
    """

    def user_annotation_split_pass(self):
        remaining_tracking_ids = set(self.model.get_all_tracking().keys())

        # display the frames
        fig, axs = plt.subplots(2, 5)

        # for tracking_id, tracking in self.model.get_all_tracking().items():
        while len(remaining_tracking_ids) > 0:

            print("Remaining tracking IDs:", remaining_tracking_ids)

            tracking_id = remaining_tracking_ids.pop()
            tracking = self.model.get_all_tracking()[tracking_id]

            fig.suptitle(f"Tracking ID: {tracking_id}")

            available_frames = list(tracking.masks.keys())

            frame_display_target = 10  # 2x5 grid
            upper_bound = max(available_frames)
            lower_bound = min(available_frames)

            separation_point = None

            while separation_point is None:

                frames_in_bound = [
                    frame_id
                    for frame_id in available_frames
                    if (lower_bound <= frame_id) and (frame_id <= upper_bound)
                ]

                selected_frames_to_display = list()
                # add frames where the tracking is not continuous
                for frame_id in frames_in_bound:
                    if frame_id - 1 not in frames_in_bound:
                        selected_frames_to_display.append(frame_id)
                    elif frame_id + 1 not in frames_in_bound:
                        selected_frames_to_display.append(frame_id)

                if len(selected_frames_to_display) > frame_display_target:
                    selected_frames_to_display = selected_frames_to_display[
                        :frame_display_target
                    ]

                remaining_frames = frame_display_target - len(
                    selected_frames_to_display
                )

                if remaining_frames > 0:
                    num_frames_pool = len(frames_in_bound)
                    sel_ids = np.linspace(
                        0, num_frames_pool - 1, remaining_frames, dtype=int
                    )
                    for sel_id in sel_ids:
                        selected_frames_to_display.append(frames_in_bound[sel_id])

                selected_frames_to_display = sorted(list(selected_frames_to_display))

                # inner loop for the user to either split the tracking or continue
                click_handler = FrameClickHandler(fig, axs)
                for i, frame_id in enumerate(selected_frames_to_display):
                    ax = axs[i // 5, i % 5]

                    frame = self.model.get_frame(frame_id)
                    decoded_mask = coco_mask.decode(tracking.masks[frame_id].encoded)

                    bbox = coco_mask.toBbox(tracking.masks[frame_id].encoded)

                    # crop the frame and the mask
                    frame = frame[
                        int(bbox[1]) : int(bbox[1] + bbox[3]),
                        int(bbox[0]) : int(bbox[0] + bbox[2]),
                    ]
                    decoded_mask = decoded_mask[
                        int(bbox[1]) : int(bbox[1] + bbox[3]),
                        int(bbox[0]) : int(bbox[0] + bbox[2]),
                    ]

                    # display the frame and the mask
                    # compute the boundary of the mask, and draw it on the frame
                    mask_boundary = cv2.Canny(decoded_mask * 255, 100, 200)
                    frame[mask_boundary > 0] = [255, 0, 0]

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    ax.imshow(frame)

                    ax.set_title(
                        f"Frame {frame_id}"
                    )

                    #set master title
                    fig.suptitle(f"Click on 2 frames containint seg point | key binding: 'n' to skip the current segmentation, 'd' to delete current tracking")

                plt.show(block=False)
                click_handler.wait_for_clicks()

                if click_handler.cancel_selection:
                    break

                if click_handler.should_delete:
                    self.model.get_all_tracking().pop(tracking_id)
                    break

                if len(click_handler.clicks) != 2:
                    continue

                sep1 = selected_frames_to_display[click_handler.clicks[0]]
                sep2 = selected_frames_to_display[click_handler.clicks[1]]

                idx1 = available_frames.index(sep1)
                idx2 = available_frames.index(sep2)

                if abs(idx2 - idx1) > 1:
                    lower_bound = min(sep1, sep2)
                    upper_bound = max(sep1, sep2)

                    continue
                else:
                    separation_point = (sep1, sep2)
                    new_id = self.model.split_tracking(tracking_id, separation_point)

                    remaining_tracking_ids.add(new_id)

        self.model.save_state()

    def user_annotation_merge_pass(self):
        """
        Given consistent tracking, the user will be asked to merge the tracking.

        For each view to be merged, a few frames from other candidate views will be shown, and the user will be asked to merge them or mark them as different.
        """

        all_trackings = self.model.get_all_tracking()
        all_tracking_ids = list(all_trackings.keys())

        possible_merges = set(combinations(all_tracking_ids, 2))

        max_candidates = 3
        example_per_candidate = 8

        fig, axs = plt.subplots(max_candidates + 1, example_per_candidate)

        while len(possible_merges) > 0:

            tracking_id_1, tracking_id_2 = next(iter(possible_merges))
            candidates = {x for x in possible_merges if tracking_id_1 in x}

            # limit the number of candidates to max_candidates
            candidates = set(list(candidates)[:max_candidates])

            # permute along the candidates to make sure the tracking_id_1 is always the first
            candidates = {
                (x[0], x[1]) if x[0] == tracking_id_1 else (x[1], x[0])
                for x in candidates
            }

            click_handler = FrameClickHandler(fig, axs, num_clicks=1)
            for ax in axs.flat:
                ax.cla()  # Clear each subplot

            # display the tracking 1
            self._display_tracking_on_axs(all_trackings[tracking_id_1], axs[0, :], is_target=True)

            for i, candidate in enumerate(candidates):
                self._display_tracking_on_axs(
                    all_trackings[candidate[1]], axs[i + 1, :]
                )

            #set master title
            fig.suptitle(f"Click on any column on the other rows that you would like to merge with the first row | key binding: 'n' to assert no merge (all candidates are different from the first one)")

            plt.show(block=False)
            click_handler.wait_for_clicks()

            if click_handler.cancel_selection:
                # no merging between the candidates.
                impossible_merges = set()
                for candidate in candidates:
                    impossible_merges.update(
                        {
                            x
                            for x in possible_merges
                            if (candidate[0] in x) and (candidate[1] in x)
                        }
                    )

                possible_merges = possible_merges - impossible_merges
                continue

            if len(click_handler.clicks) == 0:
                continue

            sel = click_handler.clicks[0] // (example_per_candidate)
            if sel == 0:
                continue

            # merge the tracking
            tracking_id_2 = list(candidates)[sel - 1][1]

            self.model.merge_tracking(tracking_id_1, tracking_id_2)

            # remove all candidates with tracking_id_2
            impossible_merges = {x for x in possible_merges if tracking_id_2 in x}
            possible_merges: Set[Tuple[int]] = possible_merges - impossible_merges

        plt.close()
        self.model.save_state()

    def _display_tracking_on_axs(self, tracking, axs, is_target=False):

        num_axs = len(axs)
        available_frames = list(tracking.masks.keys())

        # sample some frames to display
        selected_frames = np.linspace(0, len(available_frames) - 1, num_axs, dtype=int)

        selected_frames = [available_frames[i] for i in selected_frames]

        for i, frame_id in enumerate(selected_frames):

            ax = axs[i]

            frame = self.model.get_frame(frame_id)
            decoded_mask = coco_mask.decode(tracking.masks[frame_id].encoded)

            bbox = coco_mask.toBbox(tracking.masks[frame_id].encoded)

            # crop the frame and the mask
            frame = frame[
                int(bbox[1]) : int(bbox[1] + bbox[3]),
                int(bbox[0]) : int(bbox[0] + bbox[2]),
            ]
            decoded_mask = decoded_mask[
                int(bbox[1]) : int(bbox[1] + bbox[3]),
                int(bbox[0]) : int(bbox[0] + bbox[2]),
            ]

            # display the frame and the mask
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ax.imshow(decoded_mask, alpha=0.4)
            edges = cv2.Canny(decoded_mask * 255, 100, 200)
            frame[edges > 0] = [255, 0, 0]

            ax.imshow(frame)

            # disable the ticks
            if i != 0:
                ax.axis("off")

            ax.set_title(f"Frame {frame_id}")

        if is_target:
            axs[0].set_ylabel(f"Target: {tracking.tracking_id}")
        else:
            axs[0].set_ylabel(f"Candidate : {tracking.tracking_id}")


if __name__ == "__main__":

    # rgb_frames_path = "/home/inf/mvt_annotator/demo/frames"
    rgb_frames_path = "/data/people_walking"
    # annotator_states_path = "/home/inf/mvt_annotator/demo/states"  # save the states here, including all tracking, detections, masks, etc.
    annotator_states_path = "/data/states"  # save the states here, including all tracking, detections, masks, etc.

    os.makedirs(annotator_states_path, exist_ok=True)

    # Example usage
    model = SingleVideoAnnotatorModel(
        video_id="video_test",
        state_save_folder=annotator_states_path,
        video_source_path=rgb_frames_path,
        sorting_rule=lambda x: x,  # change this based on your image naming convention
    )
    controller = SingleVideoAnnotatorController(model, None)

    # connect a view to the annotator model to visualize changes
    # view = SingleVideoAnnotatorView(model, save=True)

    # model.load_state() # use this to load the state of the annotator

    model.initialize_YOLO_detection()  # this line computes the detections for the entire video
    # model.save_state()

    model.initialize_SAM_tracking()  # this one initializes SAM mask tracking for the entire video
    # model.save_state()

    # the below functions are for the user to interact with the model
    controller.user_annotation_split_pass()
    controller.user_annotation_merge_pass()

    model.save_state()

    # for rendering a video of the visualizations
    # for i in range(model.get_frame_count()):
    #     print(i)
    #     model.notify_observers(frame_id=i, changed="detections")
