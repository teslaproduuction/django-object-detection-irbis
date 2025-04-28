import os
import io
from PIL import Image as I
import torch
import collections
from ast import literal_eval
from ultralytics import YOLO

from django.views.generic.detail import DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.conf import settings
from django.shortcuts import render
from django.core.paginator import Paginator

from images.models import ImageFile
from .models import InferencedImage
from .forms import InferencedImageForm, YoloModelForm
from modelmanager.models import MLModel


class InferenceImageDetectionView(LoginRequiredMixin, DetailView):
    model = ImageFile
    template_name = "detectobj/select_inference_image.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        img_qs = self.get_object()
        imgset = img_qs.image_set
        images_qs = imgset.images.all()

        # For pagination GET request
        self.get_pagination(context, images_qs)

        if is_inf_img := InferencedImage.objects.filter(
                orig_image=img_qs
        ).exists():
            inf_img_qs = InferencedImage.objects.get(orig_image=img_qs)
            context['inf_img_qs'] = inf_img_qs

        context["img_qs"] = img_qs
        context["form1"] = YoloModelForm()
        context["form2"] = InferencedImageForm()
        return context

    def get_pagination(self, context, images_qs):
        paginator = Paginator(
            images_qs, settings.PAGINATE_DETECTION_IMAGES_NUM)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        context["is_paginated"] = (
                images_qs.count() > settings.PAGINATE_DETECTION_IMAGES_NUM
        )
        context["page_obj"] = page_obj

    def post(self, request, *args, **kwargs):
        img_qs = self.get_object()
        img_bytes = img_qs.image.read()
        img = I.open(io.BytesIO(img_bytes))

        # Get form data
        modelconf = self.request.POST.get("model_conf")
        modelconf = float(
            modelconf) if modelconf else settings.MODEL_CONFIDENCE
        custom_model_id = self.request.POST.get("custom_model")
        yolo_model_name = self.request.POST.get("yolo_model")

        # Yolov8 dirs
        yolo_weightsdir = settings.YOLOV8_WEIGTHS_DIR

        # Flag to track if we're using a custom model
        is_custom_model = False

        # Whether user selected a custom model for the detection task
        # An offline model will be used for detection provided user has
        # uploaded this model.
        if custom_model_id:
            detection_model = MLModel.objects.get(id=custom_model_id)
            model = YOLO(detection_model.pth_filepath)
            is_custom_model = True
            # Print class names for debugging
            print("Custom model class names:", model.names)

        # Whether user selected a yolo model for the detection task
        # Selected yolov8 model will be downloaded, and ready for object
        # detection task. YOLOv8 API will start working.
        elif yolo_model_name:
            model_path = os.path.join(yolo_weightsdir, yolo_model_name)
            # Download if not exists
            if not os.path.exists(model_path):
                model = YOLO(yolo_model_name)  # This will download the model
            else:
                model = YOLO(model_path)
            # Print class names for debugging
            print("YOLO model class names:", model.names)

        # If using custom model, pre-update class names before inference
        if is_custom_model:
            # Fix class names in model before running inference
            for idx in model.names:
                if model.names[idx].lower() in ['cat', 'sheep']:
                    model.names[idx] = 'irbis'
                    print(f"Updated class {idx} from 'cat'/'sheep' to 'irbis'")

        # Run inference
        results = model(img, conf=modelconf, verbose=False)

        # Process results
        results_list = []
        for r in results:
            # Update class names in results if using custom model
            if is_custom_model and hasattr(r, 'names'):
                for idx in r.names:
                    if r.names[idx].lower() in ['cat', 'sheep']:
                        r.names[idx] = 'irbis'

            detection_boxes = []
            for box in r.boxes:
                # Convert box data to dictionary
                b = box.xywhn[0].tolist()  # normalized xywh format
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Get class name (will already be updated if custom model)
                class_name = model.names[cls_id]

                # Double check and update if still cat/sheep
                if is_custom_model and class_name.lower() in ['cat', 'sheep']:
                    class_name = 'irbis'
                    model.names[cls_id] = 'irbis'

                detection_boxes.append({
                    "class": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "x": b[0],
                    "y": b[1],
                    "width": b[2],
                    "height": b[3]
                })
            results_list = detection_boxes

        # Get class occurrences
        classes_list = [item["class"] for item in results_list]
        results_counter = collections.Counter(classes_list)

        if not results_list:
            messages.warning(
                request, f'Model unable to predict. Try with another model.')
        else:
            # Create directory for inference images if it doesn't exist
            media_folder = settings.MEDIA_ROOT
            inferenced_img_dir = os.path.join(
                media_folder, "inferenced_image")
            if not os.path.exists(inferenced_img_dir):
                os.makedirs(inferenced_img_dir)

            # Make sure class names are updated in results before plotting
            if is_custom_model:
                # Update class names in each result
                for r in results:
                    if hasattr(r, 'names'):
                        for idx in r.names:
                            if r.names[idx].lower() in ['cat', 'sheep']:
                                r.names[idx] = 'irbis'

                    # Some YOLOv8 versions might store class names differently
                    if hasattr(r, 'model') and hasattr(r.model, 'names'):
                        for idx in r.model.names:
                            if r.model.names[idx].lower() in ['cat', 'sheep']:
                                r.model.names[idx] = 'irbis'

                    # Another possible location for class names
                    if hasattr(r, 'cls') and hasattr(r.cls, 'names'):
                        for idx in r.cls.names:
                            if r.cls.names[idx].lower() in ['cat', 'sheep']:
                                r.cls.names[idx] = 'irbis'

            # Save the annotated image with a unique name based on model type
            if custom_model_id:
                img_file_suffix = f"custom_{custom_model_id}"
            else:
                img_file_suffix = f"yolo_{os.path.splitext(yolo_model_name)[0]}"

            # Generate filename with model identifier to allow multiple detections of same image
            img_filename = f"{os.path.splitext(img_qs.name)[0]}_{img_file_suffix}{os.path.splitext(img_qs.name)[1]}"
            img_path = f"{inferenced_img_dir}/{img_filename}"

            # If custom model, print class names right before plotting
            if is_custom_model:
                print("Class names right before plotting:", model.names)
                print("First result names:", getattr(results[0], 'names', 'No names attribute'))

            # Save annotated image
            plotted_img = results[0].plot()
            I.fromarray(plotted_img).save(img_path, format="JPEG")

            # Create a new inference image record each time (don't use get_or_create)
            inf_img = InferencedImage(
                orig_image=img_qs,
                inf_image_path=f"{settings.MEDIA_URL}inferenced_image/{img_filename}",
            )
            inf_img.detection_info = results_list
            inf_img.model_conf = modelconf
            if custom_model_id:
                inf_img.custom_model = detection_model
            elif yolo_model_name:
                inf_img.yolo_model = yolo_model_name
            inf_img.save()

            # Get the latest inference for display
            inf_img_qs = inf_img

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # set image is_inferenced to true
        img_qs.is_inferenced = True
        img_qs.save()

        # Ready for rendering next image on same html page.
        imgset = img_qs.image_set
        images_qs = imgset.images.all()

        # For pagination POST request
        context = {}
        self.get_pagination(context, images_qs)

        context["img_qs"] = img_qs
        context["inferenced_img_dir"] = f"{settings.MEDIA_URL}inferenced_image/{img_filename}" if results_list else None
        context["results_list"] = results_list
        context["results_counter"] = results_counter
        context["form1"] = YoloModelForm()
        context["form2"] = InferencedImageForm()

        # Add the latest inferenced image to the context
        if results_list:
            context["inf_img_qs"] = inf_img_qs

        return render(self.request, self.template_name, context)