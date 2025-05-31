import os
import io
import numpy as np
from PIL import Image as I
import torch
import collections
from ast import literal_eval
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from django.views.generic.detail import DetailView
from django.views.generic import FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.conf import settings
from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from django.http import JsonResponse, HttpResponseRedirect
from django.db.models import Q

from images.models import ImageFile
from .models import InferencedImage, ManualAnnotation
from .forms import InferencedImageForm, YoloModelForm, ManualAnnotationFormSetFactory
from modelmanager.models import MLModel


class SaveAnnotationView(LoginRequiredMixin, FormView):
    """API для сохранения аннотаций через AJAX запросы."""

    def post(self, request, *args, **kwargs):
        if not request.is_ajax():
            return JsonResponse({'status': 'error', 'message': 'Only AJAX requests are allowed'}, status=400)

        image_id = request.POST.get('image_id')
        class_name = request.POST.get('class_name')
        x_center = float(request.POST.get('x_center', 0))
        y_center = float(request.POST.get('y_center', 0))
        width = float(request.POST.get('width', 0))
        height = float(request.POST.get('height', 0))

        if not image_id or not class_name:
            return JsonResponse({'status': 'error', 'message': 'Missing required data'}, status=400)

        try:
            image = ImageFile.objects.get(id=image_id)
            annotation = ManualAnnotation.objects.create(
                image=image,
                class_name=class_name,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                created_by=request.user,
                is_manual=True
            )

            # Применяем аннотации к дубликатам
            self._apply_annotations_to_duplicates(image, annotation)

            return JsonResponse({
                'status': 'success',
                'message': _('Аннотация успешно сохранена'),
                'annotation_id': annotation.id
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    def _apply_annotations_to_duplicates(self, original_image, annotation):
        """Применяет аннотацию к изображениям-дубликатам."""
        if original_image.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=original_image.image_hash
            ).exclude(id=original_image.id)

            for dup_img in duplicate_images:
                # Создаем копию аннотации для дубликата
                ManualAnnotation.objects.create(
                    image=dup_img,
                    class_name=annotation.class_name,
                    x_center=annotation.x_center,
                    y_center=annotation.y_center,
                    width=annotation.width,
                    height=annotation.height,
                    confidence=annotation.confidence,
                    created_by=annotation.created_by,
                    is_manual=True
                )


class ManualAnnotationView(LoginRequiredMixin, FormView):
    """Представление для ручной разметки изображений."""

    template_name = 'detectobj/manual_annotation.html'
    form_class = ManualAnnotationFormSetFactory

    def get_success_url(self):
        return reverse('detectobj:detection_image_detail_url', kwargs={'pk': self.image.id})

    def dispatch(self, request, *args, **kwargs):
        # Получаем изображение из URL
        self.image = get_object_or_404(ImageFile, id=self.kwargs['pk'])
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['image'] = self.image
        kwargs['user'] = self.request.user
        return kwargs

    def get_initial(self):
        # Загружаем существующие аннотации для редактирования
        initial = []

        # Получаем аннотации для текущего изображения
        for annotation in ManualAnnotation.objects.filter(image=self.image):
            initial.append({
                'class_name': annotation.class_name,
                'x_center': annotation.x_center,
                'y_center': annotation.y_center,
                'width': annotation.width,
                'height': annotation.height,
            })

        # Если нет аннотаций для текущего изображения, ищем в дубликатах
        if not initial and self.image.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=self.image.image_hash
            ).exclude(id=self.image.id)

            for dup_img in duplicate_images:
                dup_annotations = ManualAnnotation.objects.filter(image=dup_img)
                if dup_annotations.exists():
                    messages.info(
                        self.request,
                        _("Найдены аннотации с идентичного изображения из другого набора. Они загружены для редактирования.")
                    )
                    for annotation in dup_annotations:
                        initial.append({
                            'class_name': annotation.class_name,
                            'x_center': annotation.x_center,
                            'y_center': annotation.y_center,
                            'width': annotation.width,
                            'height': annotation.height,
                        })
                    break  # Берем аннотации только от первого найденного дубликата

        return initial

    def form_valid(self, form):
        # Save the manual annotations
        form.save()
        messages.success(self.request, _("Аннотации успешно сохранены"))

        # Применяем аннотации к дубликатам
        self._apply_annotations_to_duplicates()

        # Now update any existing InferencedImage records for this image
        all_annotations = ManualAnnotation.objects.filter(image=self.image)

        if all_annotations.exists():
            # Convert all manual annotations to detection format
            manual_detection_info = [annotation.to_detection_format() for annotation in all_annotations]

            # Update only inference records with custom models
            existing_inferences = InferencedImage.objects.filter(orig_image=self.image, custom_model__isnull=False)
            for inf_img in existing_inferences:
                # Only update records with custom models
                if inf_img.custom_model:
                    detection_info = inf_img.detection_info or []

                    # Filter out any existing manual annotations (to avoid duplicates)
                    model_detections = []
                    for item in detection_info:
                        # Check if this is a model detection (not a manual annotation)
                        is_manual = False
                        for annot in manual_detection_info:
                            # Compare coordinates with a small tolerance
                            if (abs(item.get('x', 0) - annot.get('x', 0)) < 0.01 and
                                    abs(item.get('y', 0) - annot.get('y', 0)) < 0.01 and
                                    abs(item.get('width', 0) - annot.get('width', 0)) < 0.01 and
                                    abs(item.get('height', 0) - annot.get('height', 0)) < 0.01):
                                is_manual = True
                                break

                        if not is_manual:
                            model_detections.append(item)

                    # Combine model detections with manual annotations
                    combined_info = model_detections + manual_detection_info

                    # Update the InferencedImage record's detection_info
                    inf_img.detection_info = combined_info

                    # Now we need to regenerate the image with annotations
                    try:
                        # Get inference image path
                        inf_img_path = inf_img.inf_image_path
                        if inf_img_path.startswith(settings.MEDIA_URL):
                            inf_img_path = inf_img_path[len(settings.MEDIA_URL):]
                        inf_img_full_path = os.path.join(settings.MEDIA_ROOT, inf_img_path)

                        # Open the original image
                        img_path = self.image.get_imagepath
                        img = I.open(img_path)

                        # Create an annotated image with YOLO
                        model = YOLO(inf_img.custom_model.pth_filepath)

                        # Run inference
                        results = model(img,
                                        conf=float(
                                            inf_img.model_conf) if inf_img.model_conf else settings.MODEL_CONFIDENCE,
                                        verbose=False)

                        # Get the first result
                        result = results[0]

                        # Plot using the result's plot method
                        plotted_img = result.plot()

                        # Create an annotator to add manual annotations
                        annotator = Annotator(plotted_img)

                        # Draw manual annotations in a different color
                        for annotation in manual_detection_info:
                            # Get normalized coordinates
                            x_center = annotation.get('x', 0)
                            y_center = annotation.get('y', 0)
                            width = annotation.get('width', 0)
                            height = annotation.get('height', 0)

                            # Convert to pixel coordinates
                            img_width, img_height = img.size
                            x1 = int((x_center - width / 2) * img_width)
                            y1 = int((y_center - height / 2) * img_height)
                            x2 = int((x_center + width / 2) * img_width)
                            y2 = int((y_center + height / 2) * img_height)

                            # Set color (manual annotations in red)
                            color = (255, 0, 0)  # RGB for red

                            # Draw the box
                            annotator.box_label([x1, y1, x2, y2], annotation.get('class', 'unknown'), color=color)

                        # Save the final image
                        I.fromarray(annotator.result()).save(inf_img_full_path, format="JPEG")

                    except Exception as e:
                        print(f"Error regenerating inference image: {e}")

                    # Save the InferencedImage record
                    inf_img.save()

            if existing_inferences.exists():
                messages.info(self.request,
                              _("Аннотации также добавлены к существующим результатам распознавания с кастомными моделями"))

        # Make sure to always redirect to the success URL
        return HttpResponseRedirect(self.get_success_url())

    def _apply_annotations_to_duplicates(self):
        """Применяет аннотации текущего изображения к его дубликатам."""
        if self.image.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=self.image.image_hash
            ).exclude(id=self.image.id)

            if duplicate_images.exists():
                all_annotations = ManualAnnotation.objects.filter(image=self.image)

                for dup_img in duplicate_images:
                    # Удаляем старые аннотации дубликата
                    ManualAnnotation.objects.filter(image=dup_img).delete()

                    # Копируем новые аннотации
                    for annotation in all_annotations:
                        ManualAnnotation.objects.create(
                            image=dup_img,
                            class_name=annotation.class_name,
                            x_center=annotation.x_center,
                            y_center=annotation.y_center,
                            width=annotation.width,
                            height=annotation.height,
                            confidence=annotation.confidence,
                            created_by=self.request.user,
                            is_manual=True
                        )

                messages.info(
                    self.request,
                    _("Аннотации также применены к %d идентичным изображениям в других наборах") % duplicate_images.count()
                )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['image'] = self.image

        # Получаем список уникальных классов из всех аннотаций
        class_names = ManualAnnotation.objects.values_list('class_name', flat=True).distinct()
        context['class_names'] = list(class_names)

        # Добавляем размеры изображения для правильной работы JavaScript
        img_width, img_height = self.image.get_imgshape
        context['img_width'] = img_width
        context['img_height'] = img_height

        # Проверяем наличие дубликатов
        if self.image.image_hash:
            duplicate_count = ImageFile.objects.filter(
                image_hash=self.image.image_hash
            ).exclude(id=self.image.id).count()

            if duplicate_count > 0:
                context['has_duplicates'] = True
                context['duplicate_count'] = duplicate_count

        return context


class InferenceImageDetectionView(LoginRequiredMixin, DetailView):
    model = ImageFile
    template_name = "detectobj/select_inference_image.html"

    def _get_all_annotations_for_image(self, img_qs):
        """Получает все аннотации для изображения, включая аннотации с дубликатов."""
        # Сначала получаем аннотации для самого изображения
        annotations = list(ManualAnnotation.objects.filter(image=img_qs))

        # Если у изображения есть хэш, ищем аннотации в дубликатах
        if img_qs.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=img_qs.image_hash
            ).exclude(id=img_qs.id)

            for dup_img in duplicate_images:
                dup_annotations = ManualAnnotation.objects.filter(image=dup_img)
                if dup_annotations.exists():
                    # Добавляем аннотации из дубликатов
                    annotations.extend(list(dup_annotations))
                    break  # Берем аннотации только от первого найденного дубликата

        # Удаляем дубликаты аннотаций по координатам
        unique_annotations = []
        seen_coords = set()

        for annotation in annotations:
            coord_key = (
                round(annotation.x_center, 3),
                round(annotation.y_center, 3),
                round(annotation.width, 3),
                round(annotation.height, 3),
                annotation.class_name
            )
            if coord_key not in seen_coords:
                seen_coords.add(coord_key)
                unique_annotations.append(annotation)

        return unique_annotations

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        img_qs = self.get_object()
        imgset = img_qs.image_set
        images_qs = imgset.images.all()

        # For pagination GET request
        self.get_pagination(context, images_qs)

        # Получаем последнюю инференс-картинку
        if is_inf_img := InferencedImage.objects.filter(
                orig_image=img_qs
        ).exists():
            inf_img_qs = InferencedImage.objects.filter(orig_image=img_qs).first()
            context['inf_img_qs'] = inf_img_qs

            # Добавляем счетчик классов, если есть результаты обнаружения
            if inf_img_qs.detection_info:
                classes_list = [item.get('class') for item in inf_img_qs.detection_info]
                context['results_counter'] = collections.Counter(classes_list)

        # Получаем ВСЕ ручные аннотации (включая дубликаты)
        manual_annotations = self._get_all_annotations_for_image(img_qs)
        context['manual_annotations'] = manual_annotations

        # Информация о дубликатах
        if img_qs.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=img_qs.image_hash
            ).exclude(id=img_qs.id)

            if duplicate_images.exists():
                context['has_duplicates'] = True
                context['duplicate_count'] = duplicate_images.count()
                context['duplicate_sets'] = [dup.image_set.name for dup in duplicate_images[:3]]  # Первые 3 набора

        # Если есть инференс, добавляем к нему ручные аннотации для отображения
        if 'inf_img_qs' in context and manual_annotations:
            inf_img = context['inf_img_qs']
            detection_info = inf_img.detection_info or []

            # Добавляем ручные аннотации к результатам детекции
            manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]

            # Обновляем информацию о результатах
            if manual_detection_info:
                combined_detection_info = detection_info + manual_detection_info
                context['all_detection_info'] = combined_detection_info

                # Обновляем счетчик классов для всех результатов
                all_classes = [item.get('class') for item in combined_detection_info]
                context['all_results_counter'] = collections.Counter(all_classes)

        # Добавляем ссылку на страницу ручной разметки
        context['manual_annotation_url'] = reverse('detectobj:manual_annotation_url', kwargs={'pk': img_qs.id})

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

        # Initialize inf_img_qs to None to avoid UnboundLocalError later
        inf_img_qs = None

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
                if is_custom_model and class_name.lower() in ['cat', 'giraffe']:
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

        # Получаем ВСЕ ручные аннотации (включая дубликаты)
        manual_annotations = self._get_all_annotations_for_image(img_qs)

        # Create directory for inference images if it doesn't exist
        media_folder = settings.MEDIA_ROOT
        inferenced_img_dir = os.path.join(
            media_folder, "inferenced_image")
        if not os.path.exists(inferenced_img_dir):
            os.makedirs(inferenced_img_dir, exist_ok=True)

        # Set default values for variables used later
        img_filename = None
        inf_img = None

        if not results_list and not (manual_annotations and is_custom_model):
            messages.warning(
                request,
                _('Модель не смогла обнаружить объекты. Попробуйте другую модель или выполните ручную разметку.'))
        else:
            # Make sure class names are updated in results before plotting
            if is_custom_model:
                # Update class names in each result
                for r in results:
                    if hasattr(r, 'names'):
                        for idx in r.names:
                            if r.names[idx].lower() in ['cat', 'giraffe']:
                                r.names[idx] = 'irbis'

                    # Some YOLOv8 versions might store class names differently
                    if hasattr(r, 'model') and hasattr(r.model, 'names'):
                        for idx in r.model.names:
                            if r.model.names[idx].lower() in ['cat', 'giraffe']:
                                r.model.names[idx] = 'irbis'

                    # Another possible location for class names
                    if hasattr(r, 'cls') and hasattr(r.cls, 'names'):
                        for idx in r.cls.names:
                            if r.cls.names[idx].lower() in ['cat', 'giraffe']:
                                r.cls.names[idx] = 'irbis'

            # Save the annotated image with a unique name based on model type
            if custom_model_id:
                img_file_suffix = f"custom_{custom_model_id}"
            else:
                img_file_suffix = f"yolo_{os.path.splitext(yolo_model_name)[0]}"

            # Generate filename with model identifier to allow multiple detections of same image
            img_filename = f"{os.path.splitext(img_qs.name)[0]}_{img_file_suffix}{os.path.splitext(img_qs.name)[1]}"
            img_path = os.path.join(inferenced_img_dir, img_filename)

            # If custom model, print class names right before plotting
            if is_custom_model:
                print("Class names right before plotting:", model.names)
                print("First result names:", getattr(results[0], 'names', 'No names attribute'))

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

            # Process manual annotations only if using a custom model
            if manual_annotations and is_custom_model:
                # Convert manual annotations to detection format
                manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]

                # Add them to the detection results
                if not results_list:
                    # If no objects were detected by the model, use only manual annotations
                    inf_img.detection_info = manual_detection_info
                else:
                    # Otherwise, combine model detections with manual annotations
                    inf_img.detection_info = results_list + manual_detection_info

                # Handle image creation with annotations
                if results_list:
                    # Get the result from inference
                    result = results[0]

                    # Plot using the result's plot method
                    plotted_img = result.plot()

                    # Create an annotator to add manual annotations
                    annotator = Annotator(plotted_img)

                    # Draw manual annotations in a different color
                    for annotation in manual_detection_info:
                        # Get normalized coordinates
                        x_center = annotation.get('x', 0)
                        y_center = annotation.get('y', 0)
                        width = annotation.get('width', 0)
                        height = annotation.get('height', 0)

                        # Convert to pixel coordinates
                        img_width, img_height = img.size
                        x1 = int((x_center - width / 2) * img_width)
                        y1 = int((y_center - height / 2) * img_height)
                        x2 = int((x_center + width / 2) * img_width)
                        y2 = int((y_center + height / 2) * img_height)

                        # Set color (manual annotations in red)
                        color = (255, 0, 0)  # RGB for red

                        # Draw the box
                        annotator.box_label([x1, y1, x2, y2], annotation.get('class', 'unknown'), color=color)

                    # Save the final image with manual annotations
                    I.fromarray(annotator.result()).save(img_path, format="JPEG")
                else:
                    # No model results, just manual annotations
                    # Create a plain image with just manual annotations
                    annotator = Annotator(np.array(img))

                    # Draw manual annotations
                    for annotation in manual_detection_info:
                        # Get normalized coordinates
                        x_center = annotation.get('x', 0)
                        y_center = annotation.get('y', 0)
                        width = annotation.get('width', 0)
                        height = annotation.get('height', 0)

                        # Convert to pixel coordinates
                        img_width, img_height = img.size
                        x1 = int((x_center - width / 2) * img_width)
                        y1 = int((y_center - height / 2) * img_height)
                        x2 = int((x_center + width / 2) * img_width)
                        y2 = int((y_center + height / 2) * img_height)

                        # Set color (manual annotations in red)
                        color = (255, 0, 0)  # RGB for red

                        # Draw the box
                        annotator.box_label([x1, y1, x2, y2], annotation.get('class', 'unknown'), color=color)

                    # Save the image with manual annotations
                    I.fromarray(annotator.result()).save(img_path, format="JPEG")
            elif results_list:
                # No manual annotations or not using custom model, just save the regular plotted image
                plotted_img = results[0].plot()
                I.fromarray(plotted_img).save(img_path, format="JPEG")

            # Save inference image with combined annotations
            inf_img.save()

            # Get the latest inference for display
            if inf_img:
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
        context["inferenced_img_dir"] = f"{settings.MEDIA_URL}inferenced_image/{img_filename}" if img_filename else None
        context["results_list"] = results_list
        context["results_counter"] = results_counter
        context["form1"] = YoloModelForm()
        context["form2"] = InferencedImageForm()

        # Add manual annotations to context
        context["manual_annotations"] = manual_annotations

        # Add link to manual annotation page
        context['manual_annotation_url'] = reverse('detectobj:manual_annotation_url', kwargs={'pk': img_qs.id})

        # Add the latest inferenced image to the context if available
        if inf_img_qs:
            context["inf_img_qs"] = inf_img_qs

        return render(self.request, self.template_name, context)